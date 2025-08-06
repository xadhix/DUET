import math
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.duet.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.utils.data_processing import split_before
from typing import Type, Dict, Optional, Tuple
from torch import optim
import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark
)
from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ...models.model_base import ModelBase, BatchMaker

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class DUET(ModelBase):
    def __init__(self, **kwargs):
        super(DUET, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len

    @property
    def model_name(self):
        return "DUET"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm"
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark
    def validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        # print("before eval")
        self.model.eval()
        # print("after eval")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # self.model.already_printed = False  # Reset the already_printed flag for each epoch     
                # print("model called for validation")           
                output, _ = self.model(input)

                target = target[:, -config.horizon:, :]
                output = output[:, -config.horizon:, :]

                # print("target shape:", target.shape)
                # output[:, :, 0] = 0.0  # Assuming the first channel is the one to be zeroed out 
                # print("hopefully real values")
                # print("target:", target)    
                # print("output:", output)

                loss = criterion(output, target).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        # print("validation training  ")
        self.model.train()
        # print("validation training finished")
        return total_loss

    def forecast_fit(self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time data data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)
        self.model = DUETModel(self.config) 
        # self.model = DUETModel(self.config, channel_names=train_valid_data.columns.tolist())  ### added channel names have to be removed later

        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len  # train_valid_data includes both train and validation data
        )

        #---------------------------------------------------------
        # # Identify the index of 'OT' in the columns

        # # channel names for Etth1 ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

        channel_names = train_data.columns.tolist()
        print(f"[DEBUG] Channel names: {channel_names}")
        # channel_to_zero = "HUFL"
        # channel_idx = channel_names.index(channel_to_zero)

        # # Set the OT channel to constant before normalization
        # print(f"[DEBUG] Setting '{channel_to_zero}' (index {channel_idx}) to constant zero.")
        # train_data.iloc[:, channel_idx] = 0.0

        # # Doing the same for validation data if it exists
        # if train_ratio_in_tv != 1:
        #     valid_data.iloc[:, channel_idx] = 0.0
        #----------------------------------------------------------

        # ========================================================
        # print("fitting train_data check for zero :", train_data.values)
        # print("fitting valid_data: check for zero :", valid_data.values)
    
        # ========================================================
        self.scaler.fit(train_data.values)
        # print("scaler mean :", self.scaler.mean_)
        # print("scaler var :", self.scaler.var_)
        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

            # print("valid_data_loader:", valid_data_loader)
            # print("valid_data:", valid_data)

        train_dataset, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        # print("train_data_loader:", train_data_loader)
        # print("train_data:", train_dataset)

        # Define the loss function and optimizer
        if config.loss == "MSE":
            criterion = nn.MSELoss()
        elif config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Check if CUDA is available and set the device accordingly
        print(f"Using device: {device}")   

        #----------------------------------------------

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            
            
            # print("forecast fit training with epoch:", epoch)

            self.model.train()

            # print("After training forecast fit epoch:", epoch)
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                    train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input

                #-----------------------------------------------
                # print("input:", input)
                # print("target:", target)
                # print("input_mark:", input_mark)
                # print("target_mark:", target_mark)
                #-----------------------------------------------
                # self.model.already_printed = False  # Reset the already_printed flag for each epoch

                output, loss_importance = self.model(input)

                target = target[:, -config.horizon:, :]
                output = output[:, -config.horizon:, :]

                #-----------------------------------------------

                # making the values zero so that it does not contribute to the loss
                # output[:, :, channel_idx] = 0.0
                # print("target shape:", target.shape)
                # print("output shape:", output.shape)
                # print("target:", target)
                # print("output:", output)

                #-----------------------------------------------

                loss = criterion(output, target)

                ############################################

                

                # print("loss:", loss)


                ##############################################

                total_loss = loss + loss_importance
                total_loss.backward()

                optimizer.step()

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, criterion)
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    break

            adjust_learning_rate(optimizer, epoch + 1, config)


        print("model probs :", len(self.model.probs))  # Print the channel mask probabilities

        # Fix: Average across batch dimension to get [1, 7, 7] for all tensors
        normalized_probs = []
        
        print("\n=== NORMALIZING ACROSS BATCH DIMENSION ===")
        for i, p in enumerate(self.model.probs):
            prob_tensor = p.cpu().detach()
            # Take mean across batch dimension: [B, 1, 7, 7] -> [1, 7, 7]
            batch_mean = torch.mean(prob_tensor, dim=0, keepdim=False)  # [1, 7, 7]
            normalized_probs.append(batch_mean.numpy())
            
            # print(f"Prob {i}: {prob_tensor.shape} -> {batch_mean.shape}")
        
        # Now all tensors have shape [1, 7, 7]
        probs_array = np.array(normalized_probs)  # Shape: [2112, 1, 7, 7]
        print(f"\nNormalized array shape: {probs_array.shape}")
        
        # Calculate mean across all 2112 tensors to get a single [1, 7, 7] or [7, 7] matrix
        mean_prob_matrix = np.mean(probs_array, axis=0)  # Shape: [1, 7, 7]
        
        # Remove the singleton dimension to get [7, 7]
        mean_prob_matrix_2d = np.squeeze(mean_prob_matrix)  # Shape: [7, 7]
        
        print(f"\nMean probability matrix shape: {mean_prob_matrix_2d.shape}")
        print("\nMean probability matrix [7x7]:")
        print(mean_prob_matrix_2d)

        #####################################


    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        print("Forecasting............................")
        """
        Make predictions.

        :param horizon: The predicted length.
        :param testdata: Time data data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.config.norm:
            train = pd.DataFrame(
                self.scaler.transform(train.values),
                columns=train.columns,
                index=train.index,
            )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        train, test = split_before(train, len(train) - config.seq_len)

        # Additional timestamp marks required to generate transformer class methods
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )

                    output, _ = self.model(input)

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon:]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:]

                output = output.cpu().numpy()[:, -config.horizon:, :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon:]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, exclude_idx: int, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # print("batch forecast eval started")
        self.model.eval()
        # print("batch forecast eval finished")
        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        #----------------------------------------------
        # ----- Make 'OT' constant before normalizing -----
        if exclude_idx is not None and exclude_idx > -1:
            # print(f"[DEBUG] Setting channel {exclude_idx} (OT) to constant zero in batch forecast.")
            input_np[:, :, exclude_idx] = 0.0
            # print("the batch forecast ", input_np)
        #----------------------------------------------

        # Store original input_np before normalization for potential return

        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)
        # print("input_np",input_np)
        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)
        # print("answers shape:", answers.shape)
        # print("answers:", answers)
        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                
                # self.model.already_printed = False  # Reset the already_printed flag for each epoch
                output, _ , = self.model(input)
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np
