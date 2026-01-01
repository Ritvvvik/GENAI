from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import os
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer
from models import GPT2Model
from dataloader import *

class Trainer:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
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

    def calc_loss(self, input_batch, target_batch, loss_fn):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = loss_fn(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def evaluate_model(self, dataloader, loss_fn):
        if len(dataloader) == 0:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for input_batch, target_batch in tqdm(dataloader):
                loss = self.calc_loss(input_batch, target_batch, loss_fn)
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
        loss_fn,
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
        self.model.train()
        for epoch in range(epochs):
            for input_batch, target_batch in tqdm(train_dataloader):

                # forward pass
                loss = self.calc_loss(input_batch, target_batch, loss_fn)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            net_epoch = last_epoch + epoch + 1
            if net_epoch % eval_freq == 0:
                train_loss = self.evaluate_model(train_dataloader, loss_fn)
                val_loss = self.evaluate_model(val_dataloader, loss_fn)
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

    # train configuration
    epochs = 2
    batch_size = 8
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    src_dir = "F:/nn/instruction-tuning/dolly"
    train_dir = os.path.join(src_dir, "tokenized_ds/train")
    val_dir = os.path.join(src_dir, "tokenized_ds/test")
    output_dir = os.path.join(src_dir, "gpt2_it")
    train_dataloader = create_data_loader(data_dir=train_dir, batch_size=batch_size, shuffle=True)
    val_dataloader = create_data_loader(data_dir=val_dir, batch_size=batch_size, shuffle=False)
      
    # train the model
    trainer = Trainer(model, tokenizer)
    trainer.train(
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        eval_freq=1,
        output_dir=output_dir,
        check_point_id=0,
        check_point_freq=1,
    )


