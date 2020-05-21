import numpy as np

from torch.optim import AdamW

from tqdm import tqdm

from teacher_model import *

from torch.optim.lr_scheduler import LambdaLR

class TeacherTrainer:
    def __init__(
        self
        , train_loader
        , val_loader
        , noise_size
        , hidden_size
        , num_layers
        , cramer_size
        , lam
        , epochs
        , critic_boost
        , start_lr
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.generator = TeacherGenerator(
            noise_size
            , hidden_size
            , num_layers
        ).to(device)

        self.critic = TeacherCritic(
            hidden_size
            , cramer_size
            , num_layers
        ).to(device)

        self.g_optimizer = AdamW(self.generator.parameters(), lr=start_lr)
        self.c_optimizer = AdamW(self.critic.parameters(), lr=start_lr)

        lr_track = np.logspace(0, -2, num=epochs)
        lr_lambda = lambda x: lr_track[x]

        self.g_sheduler = LambdaLR(self.g_optimizer, lr_lambda)
        self.c_sheduler = LambdaLR(self.c_optimizer, lr_lambda)

        self.lam = lam
        self.epochs = epochs
        self.critic_boost = critic_boost

    def train(self, test_every, save_every):
        for epoch in range(1, self.epochs+1):
            print(f"(epoch {epoch})")

            g_avg_loss = 0
            c_avg_loss = 0

            step = 0
            for (real, noised_1, noised_2, w_real, w_1, w_2) in tqdm(self.train_loader):
                step += 1

                gen_1= self.generator(noised_1)
                gen_2 = self.generator(noised_2)

                g_loss = (
                    (
                        self.critic(real, gen_2) - self.critic(gen_1, gen_2)
                    ) * w_1.unsqueeze(1) * w_2.unsqueeze(1)
                ).mean()
                self.g_optimizer.zero_grad()
                if step % self.critic_boost == 0:
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()


                alpha = torch.tensor(np.random.normal(size=real.size(0))).unsqueeze(1).float().to(device)
                blended = real * alpha + gen_1 * (1 - alpha)
                f = self.critic(blended, gen_2).mean()
                blend_grad = torch.autograd.grad(f, blended)[0]

                self.c_optimizer.zero_grad()
                c_loss = -g_loss + self.lam * ((blend_grad.norm(dim=1) - 1) ** 2).mean()
                c_loss.backward()
                self.c_optimizer.step()

                g_avg_loss += g_loss.item()
                c_avg_loss += c_loss.item()

            if epoch != self.epochs:
                self.g_sheduler.step()
                self.c_sheduler.step()
            print(f"GENERATOR LOSS: {g_avg_loss/len(self.train_loader):.5f}, CRITIC LOSS: {c_avg_loss/len(self.train_loader):.5f}")


            if epoch % save_every == 0:
                torch.save(self.generator, f"models/gen_{epoch}e_elu.pt")
                torch.save(self.critic, f"models/cri_{epoch}e_elu.pt")

            if epoch % test_every == 0:
                real_batches = []
                gen_batches = []
                w_batches = []
                with torch.no_grad():
                    for (real, noised, w) in tqdm(self.val_loader):
                        gen = self.generator(noised)

                        real_batches.append(real.detach().cpu())
                        gen_batches.append(gen.detach().cpu())
                        w_batches.append(w.detach().cpu())

                plot_distributions(
                    torch.cat(real_batches, dim=0)
                    , torch.cat(gen_batches, dim=0)
                    , torch.cat(w_batches, dim=0)
                    , epoch
                )
