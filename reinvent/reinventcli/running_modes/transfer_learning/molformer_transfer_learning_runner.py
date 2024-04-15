from time import time
import rdkit
import torch
import tqdm
import numpy as np
import os
import pandas as pd
import json

from reinvent_models.molformer.dataset.preprocessing import get_pair_generator
import reinvent_models.molformer.dataset.paired_dataset as molformer_dataset
from reinvent_models.reinvent_core.models.model import Model
from reinvent_chemistry.similarity import Similarity
from reinvent_chemistry.conversions import Conversions

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations.transfer_learning.molformer_transfer_learning_configuration import (
    MolformerTransferLearningConfiguration,
)
from running_modes.transfer_learning.logging.base_transfer_learning_logger import (
    BaseTransferLearningLogger,
)

from running_modes.transfer_learning.optimizers import NoamOpt


import rdkit.Chem as rkc
from rdkit.Chem import AllChem
from rdkit import DataStructs

rdkit.rdBase.DisableLog("rdApp.error")


class MolformerTransferLearningRunner(BaseRunningMode):
    """Trains a given model."""

    def __init__(
        self,
        model: Model,
        configuration: MolformerTransferLearningConfiguration,
        logger: BaseTransferLearningLogger,
        optimizer: NoamOpt,
    ):
        self._model = model
        self._configuration = configuration
        self._logger = logger
        self._optimizer = optimizer

        self.device = self._model.device
        self.similarity = Similarity()
        self.conversions = Conversions()

    def run(self):
        smiles_path = self._configuration.input_smiles_path
        validation_percentage = self._configuration.validation_percentage
        ranking_loss_penalty = self._configuration.ranking_loss_penalty

        pairs = self._generate_pairs(smiles_path)
        if len(pairs) == 0:
            raise IndexError("No suitable pairs found in the list provided. Please use the file with similar molecules.") 
        source_smiles = pairs["Source_Mol"].drop_duplicates().values

        log_data = pd.DataFrame(
            [[smi, smi] for smi in source_smiles], columns=["Source_Mol", "Target_Mol"]
        )
        log_dataloader = self._initialize_dataloader(log_data, shuffle=False)

        time_fp = time()
        if validation_percentage > 0:
            # FIXME: make the following code more readable
            idx = np.arange(len(source_smiles))

            if self._configuration.validation_seed is not None:
                np.random.seed(self._configuration.validation_seed)
            np.random.shuffle(idx)
            n_validation = int(np.ceil(len(source_smiles) * validation_percentage))
            validation_mask = np.zeros(len(pairs), dtype=np.bool)
            validation_smiles = set(source_smiles[idx[:n_validation]])
            for i, smi in enumerate(pairs["Source_Mol"]):
                validation_mask[i] = smi in validation_smiles
            train_pairs = pairs.loc[~validation_mask].reset_index(drop=True)
            validation_pairs = pairs.loc[validation_mask].reset_index(drop=True)
            self._logger.log_message(f"Indexing training pairs...")
            dataloader = self._initialize_dataloader(train_pairs)
            self._logger.log_message(f"Indexing validation pairs...")
            validation_dataloader = self._initialize_dataloader(validation_pairs)

            train_pairs.to_csv(
                os.path.join(
                    os.path.dirname(self._configuration.output_model_path),
                    f"train_{time_fp:f}.csv",
                ),
                index=False,
            )
            validation_pairs.to_csv(
                os.path.join(
                    os.path.dirname(self._configuration.output_model_path),
                    f"validation_{time_fp:f}.csv",
                ),
                index=False,
            )
            self._logger.log_message(f"Train Pairs: {len(train_pairs):d}")
            self._logger.log_message(f"Validation Pairs: {len(validation_pairs):d}")
        else:
            train_pairs = pairs
            self._logger.log_message(f"Indexing training pairs...")
            dataloader = self._initialize_dataloader(train_pairs)
            validation_dataloader = None
            train_pairs.to_csv(
                os.path.join(
                    os.path.dirname(self._configuration.output_model_path), "train.csv"
                ),
                index=False,
            )
            self._logger.log_message(f"Train Pairs: {len(train_pairs):d}")

        last_epoch = (
            self._configuration.starting_epoch + self._configuration.num_epochs - 1
        )
        losses_log = {"train": {}, "validation": {}}
        for epoch in range(self._configuration.starting_epoch, last_epoch + 1):
            self._model.set_mode("training")
            loss_epoch_train, training_nlls = self._train_epoch(epoch, dataloader)
            losses_log["train"][epoch] = loss_epoch_train
            # TODO: it is maybe more useful to use TensorBoard or similar
            self._logger.log_message(f"[TRAIN] Loss={loss_epoch_train:.3f}")
            if (self._configuration.save_every_n_epochs == 0) or (
                (epoch % self._configuration.save_every_n_epochs) == 0
            ):
                model_path = self._save_model(epoch)
                self._send_log_timestep(
                    epoch, model_path, training_nlls, log_dataloader
                )

            if validation_dataloader:
                self._model.set_mode("inference")
                # (
                #    loss_epoch_validation,
                #    accuracy,
                #    token_accuracy,
                #    similarities,
                # ) = self._compute_stats(validation_dataloader)
                loss_epoch_validation = self._compute_stats(validation_dataloader)
                losses_log["validation"][epoch] = loss_epoch_validation

                # TODO: it is maybe more useful to use TensorBoard or similar
                msg = f"[VALIDATION] Loss={loss_epoch_validation:.3f}"
                # msg += f"Molecule Acc={accuracy:.3f} - "
                # msg += f"Token Acc={token_accuracy:.3f} - "
                # msg += f"Similarities={similarities:.3f}"

                self._logger.log_message(msg)

        loss_path = os.path.join(
            os.path.dirname(self._configuration.output_model_path), "losses.json"
        )

        with open(loss_path, "w") as jloss:
            json.dump(losses_log, jloss)

    def _send_log_timestep(self, epoch, model_path, training_nlls, log_dataloader):
        self._model.set_mode("inference")
        sampled_smiles, sampled_nlls = [], []
        for batch in log_dataloader:
            src, src_mask, trg, trg_mask, sim = (
                batch.input,
                batch.input_mask,
                batch.output,
                batch.output_mask,
                batch.tanimoto,
            )
            sampled_smiles_dto = self._model.sample(src, src_mask, "multinomial")
            for ssd in sampled_smiles_dto:
                sampled_smiles.append(ssd.output)
                sampled_nlls.append(ssd.nll)

        sampled_smiles = np.array(sampled_smiles)
        sampled_nlls = np.array(sampled_nlls)
        lr = 0.0
        for param_group in self._optimizer.optimizer.param_groups:
            if "lr" in param_group:
                lr = param_group["lr"]
                break

        jsd_data_placeholder = {
            "sampled.validation": 0.0,
            "sampled.training": 0.0,
            "training.validation": 0.0,
        }

        self._logger.log_timestep(
            lr=lr,
            epoch=epoch,
            sampled_smiles=sampled_smiles,
            sampled_nlls=sampled_nlls,
            validation_nlls=None,  # not used
            training_nlls=training_nlls,
            jsd_data=jsd_data_placeholder,  # not used
            jsd_joined_data=0.0,  # not used
            model=None,  # not used
            model_path=model_path,
        )

    def _train_epoch(self, epoch, dataloader):
        pad = 0  # TODO: pad = self._model.vocabulary._padding_value
        total_loss = 0.0
        total_examples = 0.0
        use_ranking_loss = self._configuration.ranking_loss_penalty
        losses = []
        for _, batch in enumerate(
            self._progress_bar(dataloader, total=len(dataloader))
        ):
            src, src_mask, trg, trg_mask, sim = (
                batch.input,
                batch.input_mask,
                batch.output,
                batch.output_mask,
                batch.tanimoto,
            )

            self._optimizer.optimizer.zero_grad()
            if src.device != self.device:
                src = src.to(self.device)
            if src_mask.device != self.device:
                src_mask = src_mask.to(self.device)
            if trg.device != self.device:
                trg = trg.to(self.device)
            if trg_mask.device != self.device:
                trg_mask = trg_mask.to(self.device)

            nll = self._calculate_loss(src, src_mask, trg, trg_mask)
            if use_ranking_loss:
                nll = nll.reshape(dataloader.batch_size, -1)
                sim = sim.reshape(dataloader.batch_size, -1)
                y = 2.0 * (sim[..., None] > sim[:, None]) - 1.0
                ranking_loss = torch.maximum(
                    torch.zeros_like(y), y * (nll[..., None] - nll[:, None])
                )
                ranking_loss = ranking_loss.sum((-2, -1)) / (
                    sim.shape[-1] * (sim.shape[-1] - 1)
                )

                loss = nll.mean() + ranking_loss.mean()
            else:
                loss = nll.mean()
            loss.backward()
            self._optimizer.step()

            if len(losses) * len(trg) < 10000:
                losses.append(nll.detach().cpu().numpy())

            total_examples += len(trg)
            total_loss += float(loss.detach().cpu().numpy()) * len(trg)

            # if total_examples >= 100000:
            #    break

        loss_epoch = total_loss / total_examples
        return loss_epoch, np.concatenate(losses)

    def _progress_bar(self, iterable, total, **kwargs):
        return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

    def _generate_pairs(self, path):
        smiles = pd.read_csv(path, sep=",")
        pair_config = self._configuration.pairs
        pair_generator = get_pair_generator(pair_config["type"], **pair_config)
        pairs = pair_generator.build_pairs(smiles, processes=os.cpu_count())
        if len(pairs) == 0:
            raise IOError(f"No valid entries are present in the supplied file: {path}")
        return pairs

    def _initialize_dataloader(self, data, *, shuffle=None):
        if shuffle is None:
            shuffle = self._configuration.shuffle_each_epoch

        paired_dataset_args = {
            "smiles_input": data["Source_Mol"],
            "smiles_output": data["Target_Mol"],
            "vocabulary": self._model.vocabulary,
            "tokenizer": self._model.tokenizer,
        }

        if self._configuration.ranking_loss_penalty:
            paired_dataset_args["target_per_source"] = self._configuration.pairs[
                "target_per_source"
            ]
            dataset = molformer_dataset.StratifiedPairedDataset(**paired_dataset_args)
            collate_fn = molformer_dataset.StratifiedPairedDataset.collate_fn
        else:
            dataset = molformer_dataset.PairedDataset(**paired_dataset_args)
            collate_fn = molformer_dataset.PairedDataset.collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._configuration.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        return dataloader

    def _calculate_loss(self, src, src_mask, trg, trg_mask):
        nll = self._model.likelihood(src, src_mask, trg, trg_mask)
        return nll

    def _save_model(self, epoch):
        self._model.save_to_file(self._model_path(epoch))
        torch.save(self._optimizer.save_state_dict(), self._model_path(epoch) + ".opt")
        return self._model_path(epoch)

    def _model_path(self, epoch):
        path = (
            f"{self._configuration.output_model_path}.{epoch}"
            if epoch != self._configuration.num_epochs
            else f"{self._configuration.output_model_path}"
        )
        return path

    def validation_stat(
        self, dataloader, model, loss_compute, device, vocab, without_property=False
    ):
        pass

    def _compute_stats(self, dataloader):
        pad = 0  # TODO: pad = self._model.vocabulary._padding_value
        total_loss = 0
        n_correct = 0
        total_tokens = 0
        total_examples = 0
        n_correct_token = 0
        similarities = []
        vocabulary = self._model.vocabulary
        tokenizer = self._model.tokenizer
        for i, batch in enumerate(
            self._progress_bar(dataloader, total=len(dataloader))
        ):
            src, src_mask, trg, trg_mask = (
                batch.input,
                batch.input_mask,
                batch.output,
                batch.output_mask,
            )
            src = src.to(self.device)
            src_mask = src_mask.to(self.device)
            trg = trg.to(self.device)
            trg_mask = trg_mask.to(self.device)
            with torch.no_grad():

                # number of tokens without padding
                ntokens = float((trg != pad).detach().cpu().sum())

                loss = self._calculate_loss(src, src_mask, trg, trg_mask)
                total_tokens += ntokens
                total_examples += len(trg)
                total_loss += float(loss.mean().detach().cpu().numpy()) * len(trg)

        #                smiles = self._model.sample(src, src_mask, "greedy")
        #                # Compute accuracy
        #                for j in range(trg.size()[0]):
        #                    source_seq = tokenizer.untokenize(
        #                        vocabulary.decode(src[j].cpu().numpy())
        #                    )
        #                    seq = smiles[j].output
        #                    target = trg[j]
        #                    target = tokenizer.untokenize(
        #                        vocabulary.decode(target.cpu().numpy())
        #                    )
        #
        #                    # molecular accuracy
        #                    if seq == target:
        #                        n_correct += 1
        #                    # token accuracy
        #                    for k in range(len(target)):
        #                        if k < len(seq) and seq[k] == target[k]:
        #                            n_correct_token += 1
        #
        #                    # tanimoto similarity
        #                    mol1 = self.conversions.smiles_to_fingerprints([seq])
        #                    mol2 = self.conversions.smiles_to_fingerprints([source_seq])
        #                    mol1 = mol1[0] if len(mol1) else None
        #                    mol2 = mol2[0] if len(mol2) else None
        #                    if mol1 and mol2:
        #                        sim = self.similarity.calculate_tanimoto([mol1], [mol2])
        #                        sim = float(sim.ravel())
        #                    else:
        #                        sim = 0.0
        #                    similarities.append(sim)

        loss_epoch = total_loss / total_examples
        return loss_epoch


#        accuracy = n_correct * 1.0 / total_examples
#        token_accuracy = n_correct_token / total_tokens

#        sim_avg = 0
#        if len(similarities) > 0:
#            sim_avg = sum(similarities) / len(similarities)
#        return loss_epoch, accuracy, token_accuracy, sim_avg
