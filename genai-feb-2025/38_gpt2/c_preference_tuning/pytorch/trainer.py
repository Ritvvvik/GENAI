from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import os
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer
from models import GPT2Model
from dataloader import *
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, ref_model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.ref_model = ref_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plot_losses(self, epochs_seen, train_losses, val_losses, plot_dir):
        fig, ax1 = plt.subplots(figsize=(10, 10))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, "plots.pdf"))
        # plt.show()

    def calc_loss(
        self,
        chosen_input_batch,
        chosen_target_batch,
        rej_input_batch,
        rej_target_batch,
        beta,
    ):
        chosen_input_batch = chosen_input_batch.to(self.device)
        chosen_target_batch = chosen_target_batch.to(self.device)
        rej_input_batch = rej_input_batch.to(self.device)
        rej_target_batch = rej_target_batch.to(self.device)

        model_chosen_logits = self.model(chosen_input_batch)
        model_chosen_log_probs = F.cross_entropy(
            model_chosen_logits.flatten(0, 1), chosen_target_batch.flatten()
        )

        model_rej_logits = self.model(rej_input_batch)
        model_rej_log_probs = F.cross_entropy(
            model_rej_logits.flatten(0, 1), rej_target_batch.flatten()
        )

        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_input_batch)
            ref_chosen_log_probs = F.cross_entropy(
                ref_chosen_logits.flatten(0, 1), chosen_target_batch.flatten()
            )

            ref_rej_logits = self.ref_model(rej_input_batch)
            ref_rej_log_probs = F.cross_entropy(
                ref_rej_logits.flatten(0, 1), rej_target_batch.flatten()
            )

        model_log_ratios = model_chosen_log_probs - model_rej_log_probs
        ref_log_ratios = ref_chosen_log_probs - ref_rej_log_probs
        logits = model_log_ratios - ref_log_ratios
        return -F.logsigmoid(beta * logits)

    def evaluate_model(self, dataloader, beta):
        if len(dataloader) == 0:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for (
                chosen_input_batch,
                chosen_target_batch,
                rej_input_batch,
                rej_target_batch,
                ) in tqdm(dataloader):
                loss = self.calc_loss(
                    chosen_input_batch,
                    chosen_target_batch,
                    rej_input_batch,
                    rej_target_batch,
                    beta,
                )
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(dataloader)

    def clear_check_points(self, output_dir):
        check_point_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(check_point_dir):
            shutil.rmtree(check_point_dir)

    def save_check_point(self, output_dir, net_epoch, train_losses, val_losses):
        sub_id = "final" if net_epoch == -1 else str(net_epoch)
        meta_data = {
            "last_epoch": net_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        check_point_subdir = os.path.join(output_dir, "check_points", sub_id)
        if not os.path.exists(check_point_subdir):
            os.makedirs(check_point_subdir)

        print(f"Saving checkpoints to {check_point_subdir}")
        torch.save(self.model, os.path.join(check_point_subdir, "model.pkl"))
        torch.save(self.tokenizer, os.path.join(check_point_subdir, "tokenizer.pkl"))
        torch.save(meta_data, os.path.join(check_point_subdir, "meta_data.pkl"))

    def load_check_point_data(self, output_dir, check_point_id):
        check_point_subdir = os.path.join(
            output_dir, "check_points", str(check_point_id)
        )
        if not os.path.exists(check_point_subdir):
            print(f"Output irectory {check_point_subdir} does not exist.")
            return

        self.model = torch.load(
            os.path.join(check_point_subdir, "model.pkl"), weights_only=False
        )
        self.tokenizer = torch.load(
            os.path.join(check_point_subdir, "tokenizer.pkl"), weights_only=False
        )
        meta_data = torch.load(
            os.path.join(check_point_subdir, "meta_data.pkl"), weights_only=False
        )
        return meta_data

    def train(
        self,
        optimizer,
        beta,
        epochs,
        train_dataloader,
        val_dataloader,
        eval_freq,
        output_dir,
        check_point_id=0,
        check_point_freq=5,
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        check_point_dir = os.path.join(output_dir, "check_points")
        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)
        plots_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)


        print(">>> Begin Training...")
        if check_point_id != 0:
            meta_data = self.load_check_point_data(output_dir, check_point_id)
            train_losses = meta_data["train_losses"]
            val_losses = meta_data["val_losses"]
            last_epoch = meta_data["last_epoch"]
        else:
            self.clear_check_points(output_dir)
            last_epoch = 0
            train_losses, val_losses = [], []
        self.model.to(self.device)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        self.model.train()
        for epoch in range(epochs):
            for (
                chosen_input_batch,
                chosen_target_batch,
                rej_input_batch,
                rej_target_batch,
            ) in tqdm(train_dataloader):

                # forward pass
                loss = self.calc_loss(
                    chosen_input_batch,
                    chosen_target_batch,
                    rej_input_batch,
                    rej_target_batch,
                    beta,
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            net_epoch = last_epoch + epoch + 1
            if net_epoch % eval_freq == 0:
                train_loss = self.evaluate_model(train_dataloader, beta)
                val_loss = self.evaluate_model(val_dataloader, beta)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Epoch {net_epoch}: "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
            if net_epoch % check_point_freq == 0:
                self.save_check_point(output_dir, net_epoch, train_losses, val_losses)
        self.save_check_point(output_dir, -1, train_losses, val_losses)
        print(">>> End Training...")

        print(">>> Begin Plot Creation...")
        epochs_tensor = torch.linspace(0, net_epoch, len(train_losses))
        self.plot_losses(epochs_tensor, train_losses, val_losses, plots_dir)
        print(">>> End Plot Creation...")

if __name__ == "__main__":
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens(
        {
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": tokenizer.eos_token,
        }
    )
    # model
    model = GPT2Model.from_pretrained("gpt2", len(tokenizer))
    ref_model = GPT2Model.from_pretrained("gpt2", len(tokenizer))

    # train configuration
    epochs = 2
    batch_size = 8
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    src_dir = "F:/nn/preference_tuning/orca"
    train_dir = os.path.join(src_dir, "tokenized_ds/train")
    val_dir = os.path.join(src_dir, "tokenized_ds/test")
    output_dir = os.path.join(src_dir, "gpt2_it_final")
    train_dataloader = create_data_loader(data_dir=train_dir, batch_size=batch_size, shuffle=True)
    val_dataloader = create_data_loader(data_dir=val_dir, batch_size=batch_size, shuffle=False)
    

    # train the model
    trainer = Trainer(model, ref_model, tokenizer)
    trainer.train(
        optimizer=optimizer,
        beta=0.1,
        epochs=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        eval_freq=1,
        output_dir=output_dir,
        check_point_id=0,
        check_point_freq=1,
    )


