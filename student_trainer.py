import os
import time

import numpy as np

from torch.optim import AdamW

from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm

#from teacher_model import *
from student_model import *

class StudentTrainerMimic:
    def __init__(
        self
        , train_loader
        , val_loader
        , val_loader_unit_batch
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , start_lr
        , teacher_generator
        , teacher_mimic_layer
        , student_mimic_layer
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_unit_batch = val_loader_unit_batch

        self.student_num_layers = student_num_layers

        self.teacher_mimic_layer = teacher_mimic_layer
        self.student_mimic_layer = student_mimic_layer

        self.teacher_generator = teacher_generator
        self.teacher_generator.dnn[teacher_mimic_layer].register_forward_hook(
            save_output_of_layer
        )

        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
            , 128
        ).to(device)

        self.student_generator.dnn[student_mimic_layer].register_forward_hook(
            save_output_of_layer
        )

        self.start_lr = start_lr

        print(self.teacher_generator)
        print(self.student_generator)

        self.mimic_epochs = epochs // 2
        self.train_epochs = epochs - self.mimic_epochs

    def train(self, metric_every, savename, val_size=None):
        if not val_size:
            val_size = len(self.val_loader)

        if os.path.isfile(f"results/{savename}.txt"):
            os.remove(f"results/{savename}.txt")

        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.mimic_epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.mimic_epochs+1):
            print(f"epoch {epoch} (mimic)")

            avg_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                mimic = self.student_generator.mimic(
                    self.student_generator.dnn[self.student_mimic_layer].saved
                )

                loss = ((mimic - self.teacher_generator.dnn[self.teacher_mimic_layer].saved) ** 2).mean()

                avg_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.mimic_epochs:
                self.sheduler.step()
            print(f"mimic loss: {avg_loss/len(self.train_loader):.6f}")


        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.mimic_epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.train_epochs+1):
            print(f"epoch {epoch} (train)")

            avg_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                loss = ((teacher_generated - student_generated) ** 2).mean()

                avg_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.train_epochs:
                self.sheduler.step()
            avg_loss = avg_loss/len(self.train_loader)
            print(f"train loss: {avg_loss:.6f}")

            if epoch % metric_every == 0:
                self.validate(val_size, epoch, savename, avg_loss)

    def validate(self, val_size, epoch, savename, avg_loss):
        with torch.no_grad():
            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.teacher_generator(noised)
                n += 1
                if n == 1000:
                    break
            teacher_time = (time.time() - start) / 1000

            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.student_generator(noised)
                n += 1
                if n == 1000:
                    break
            student_time = (time.time() - start) / 1000

            teacher_ms = teacher_time * 1000
            student_ms = student_time * 1000

            print(f"avg teacher: {teacher_ms:.3f}ms, avg student: {student_ms:.3f}ms")

            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            step = 0
            for (real, noised, w) in self.val_loader:
                step += 1
                if step > val_size:
                    break
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

        info = {
            "teacher time, ms": teacher_ms
            , "student time, ms": student_ms
            , "loss": avg_loss
            , "epoch": epoch
        }

        plot_metrics(
            torch.cat(real_batches, dim=0)
            , torch.cat(teacher_gen_batches, dim=0)
            , torch.cat(student_gen_batches, dim=0)
            , torch.cat(w_batches, dim=0)
            , info
            , savename
        )


class StudentTrainerIdle:
    def __init__(
        self
        , train_loader
        , val_loader
        , val_loader_unit_batch
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , start_lr
        , teacher_generator
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_unit_batch = val_loader_unit_batch

        self.student_num_layers = student_num_layers

        self.teacher_generator = teacher_generator
        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
            , 128
        ).to(device)

        self.start_lr = start_lr

        print(self.teacher_generator)
        print(self.student_generator)

        self.epochs = epochs

    def train(self, metric_every, savename, val_size=None):
        if not val_size:
            val_size = len(self.val_loader)

        if os.path.isfile(f"results/{savename}.txt"):
            os.remove(f"results/{savename}.txt")

        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.epochs+1):
            print(f"epoch {epoch} (idle)")

            avg_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                loss = ((teacher_generated - student_generated) ** 2).mean()

                avg_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.epochs:
                self.sheduler.step()
            avg_loss = avg_loss/len(self.train_loader)
            print(f"train loss: {avg_loss:.6f}")

            if epoch % metric_every == 0:
                self.validate(val_size, epoch, savename, avg_loss)

    def validate(self, val_size, epoch, savename, avg_loss):
        with torch.no_grad():
            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.teacher_generator(noised)
                n += 1
                if n == 1000:
                    break
            teacher_time = (time.time() - start) / 1000

            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.student_generator(noised)
                n += 1
                if n == 1000:
                    break
            student_time = (time.time() - start) / 1000

            teacher_ms = teacher_time * 1000
            student_ms = student_time * 1000

            print(f"avg teacher: {teacher_ms:.3f}ms, avg student: {student_ms:.3f}ms")

            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            step = 0
            for (real, noised, w) in self.val_loader:
                step += 1
                if step > val_size:
                    break
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

        info = {
            "teacher time, ms": teacher_ms
            , "student time, ms": student_ms
            , "loss": avg_loss
            , "epoch": epoch
        }

        plot_metrics(
            torch.cat(real_batches, dim=0)
            , torch.cat(teacher_gen_batches, dim=0)
            , torch.cat(student_gen_batches, dim=0)
            , torch.cat(w_batches, dim=0)
            , info
            , savename
        )


class StudentTrainerSigma:
    def __init__(
        self
        , train_loader
        , val_loader
        , val_loader_unit_batch
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , start_lr
        , teacher_generator
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_unit_batch = val_loader_unit_batch

        self.student_num_layers = student_num_layers

        self.teacher_generator = teacher_generator
        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
            , 128
        ).to(device)

        self.start_lr = start_lr

        print(self.teacher_generator)
        print(self.student_generator)

        self.epochs = epochs

    def train(self, metric_every, savename, sigma, val_size=None):
        if not val_size:
            val_size = len(self.val_loader)

        if os.path.isfile(f"results/{savename}.txt"):
            os.remove(f"results/{savename}.txt")

        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.epochs+1):
            print(f"epoch {epoch} (idle)")

            avg_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                disturbance = torch.tensor(
                    np.random.normal(loc=0.0, scale=sigma, size=teacher_generated.size())
                ).to(device)

                teacher_generated = (1 + disturbance) * teacher_generated

                loss = ((teacher_generated - student_generated) ** 2).mean()

                avg_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.epochs:
                self.sheduler.step()
            avg_loss = avg_loss/len(self.train_loader)
            print(f"train loss: {avg_loss:.6f}")

            if epoch % metric_every == 0:
                self.validate(val_size, epoch, savename, avg_loss)

    def validate(self, val_size, epoch, savename, avg_loss):
        with torch.no_grad():
            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.teacher_generator(noised)
                n += 1
                if n == 1000:
                    break
            teacher_time = (time.time() - start) / 1000

            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.student_generator(noised)
                n += 1
                if n == 1000:
                    break
            student_time = (time.time() - start) / 1000

            teacher_ms = teacher_time * 1000
            student_ms = student_time * 1000

            print(f"avg teacher: {teacher_ms:.3f}ms, avg student: {student_ms:.3f}ms")

            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            step = 0
            for (real, noised, w) in self.val_loader:
                step += 1
                if step > val_size:
                    break
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

        info = {
            "teacher time, ms": teacher_ms
            , "student time, ms": student_ms
            , "loss": avg_loss
            , "epoch": epoch
        }

        plot_metrics(
            torch.cat(real_batches, dim=0)
            , torch.cat(teacher_gen_batches, dim=0)
            , torch.cat(student_gen_batches, dim=0)
            , torch.cat(w_batches, dim=0)
            , info
            , savename
        )


class StudentTrainerRelational:
    def __init__(
        self
        , train_loader
        , val_loader
        , val_loader_unit_batch
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , start_lr
        , teacher_generator
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_unit_batch = val_loader_unit_batch

        self.student_num_layers = student_num_layers

        self.teacher_generator = teacher_generator
        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
            , 128
        ).to(device)

        self.start_lr = start_lr

        print(self.teacher_generator)
        print(self.student_generator)

        self.epochs = epochs

    def train(self, metric_every, savename, lam=1.0, val_size=None):
        if not val_size:
            val_size = len(self.val_loader)

        if os.path.isfile(f"results/{savename}.txt"):
            os.remove(f"results/{savename}.txt")

        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.epochs+1):
            print(f"epoch {epoch} (relational)")

            avg_main_loss = 0
            avg_relation_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                batch_size = noised.size(0)
                half_batch = batch_size // 2

                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                main_loss = ((teacher_generated - student_generated) ** 2).mean()

                # Relaional loss

                s_up = student_generated[:half_batch]
                s_down = student_generated[half_batch:half_batch * 2]

                t_up = teacher_generated[:half_batch]
                t_down = teacher_generated[half_batch:half_batch * 2]

                t_distances = (((t_up - t_down) ** 2).sum(axis=1)) ** (1/2)
                s_distances = (((s_up - s_down) ** 2).sum(axis=1)) ** (1/2)

                mu = t_distances.mean()

                t_potentials = t_distances / mu
                s_potentials = s_distances / mu

                relation_loss = ((t_potentials - s_potentials) ** 2).mean()

                #

                avg_main_loss += main_loss
                avg_relation_loss += relation_loss

                loss = main_loss + lam * relation_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.epochs:
                self.sheduler.step()
            avg_main_loss = avg_main_loss/len(self.train_loader)
            avg_relation_loss = avg_relation_loss/len(self.train_loader)
            print(f"main loss: {avg_main_loss:.6f} relation loss: {avg_relation_loss:.6f}")

            if epoch % metric_every == 0:
                self.validate(val_size, epoch, savename, avg_main_loss, avg_relation_loss)

    def validate(self, val_size, epoch, savename, avg_main_loss, avg_relation_loss):
        with torch.no_grad():
            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.teacher_generator(noised)
                n += 1
                if n == 1000:
                    break
            teacher_time = (time.time() - start) / 1000

            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.student_generator(noised)
                n += 1
                if n == 1000:
                    break
            student_time = (time.time() - start) / 1000

            teacher_ms = teacher_time * 1000
            student_ms = student_time * 1000

            print(f"avg teacher: {teacher_ms:.3f}ms, avg student: {student_ms:.3f}ms")

            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            step = 0
            for (real, noised, w) in self.val_loader:
                step += 1
                if step > val_size:
                    break
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

        info = {
            "teacher time, ms": teacher_ms
            , "student time, ms": student_ms
            , "main loss": avg_main_loss
            , "relation loss": avg_relation_loss
            , "epoch": epoch
        }

        plot_metrics(
            torch.cat(real_batches, dim=0)
            , torch.cat(teacher_gen_batches, dim=0)
            , torch.cat(student_gen_batches, dim=0)
            , torch.cat(w_batches, dim=0)
            , info
            , savename
        )


class StudentTrainerUltimate:
    def __init__(
        self
        , train_loader
        , val_loader
        , val_loader_unit_batch
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , start_lr
        , teacher_generator
        , teacher_mimic_layer
        , student_mimic_layer
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_unit_batch = val_loader_unit_batch

        self.student_num_layers = student_num_layers

        self.teacher_mimic_layer = teacher_mimic_layer
        self.student_mimic_layer = student_mimic_layer

        self.teacher_generator = teacher_generator
        self.teacher_generator.dnn[teacher_mimic_layer].register_forward_hook(
            save_output_of_layer
        )

        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
            , 128
        ).to(device)

        self.student_generator.dnn[student_mimic_layer].register_forward_hook(
            save_output_of_layer
        )

        self.start_lr = start_lr

        print(self.teacher_generator)
        print(self.student_generator)

        self.mimic_epochs = 40
        self.train_epochs = epochs

    def train(self, metric_every, savename, lam=1.0, sigma=0.001, val_size=None):
        if not val_size:
            val_size = len(self.val_loader)

        if os.path.isfile(f"results/{savename}.txt"):
            os.remove(f"results/{savename}.txt")

        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.mimic_epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.mimic_epochs+1):
            print(f"epoch {epoch} (mimic)")

            avg_mimic_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                mimic = self.student_generator.mimic(
                    self.student_generator.dnn[self.student_mimic_layer].saved
                )

                mimic_loss = ((mimic - self.teacher_generator.dnn[self.teacher_mimic_layer].saved) ** 2).mean()

                avg_mimic_loss += mimic_loss

                loss = mimic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.mimic_epochs:
                self.sheduler.step()
            avg_mimic_loss = avg_mimic_loss/len(self.train_loader)
            print(f"mimic loss: {avg_mimic_loss:.6f}")


        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.train_epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.train_epochs+1):
            print(f"epoch {epoch} (train)")

            avg_main_loss = 0
            avg_relation_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                batch_size = noised.size(0)
                half_batch = batch_size // 2

                disturbance = torch.tensor(
                    np.random.normal(loc=0.0, scale=sigma, size=teacher_generated.size())
                ).to(device)

                teacher_generated = (1 + disturbance) * teacher_generated

                main_loss = ((teacher_generated - student_generated) ** 2).mean()

                # Relaional loss

                s_up = student_generated[:half_batch]
                s_down = student_generated[half_batch:half_batch * 2]

                t_up = teacher_generated[:half_batch]
                t_down = teacher_generated[half_batch:half_batch * 2]

                t_distances = (((t_up - t_down) ** 2).sum(axis=1)) ** (1/2)
                s_distances = (((s_up - s_down) ** 2).sum(axis=1)) ** (1/2)

                mu = t_distances.mean()

                t_potentials = t_distances / mu
                s_potentials = s_distances / mu

                relation_loss = ((t_potentials - s_potentials) ** 2).mean()
                #

                avg_main_loss += main_loss
                avg_relation_loss += relation_loss

                loss = main_loss + lam * relation_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.train_epochs:
                self.sheduler.step()
            avg_main_loss = avg_main_loss/len(self.train_loader)
            avg_relation_loss = avg_relation_loss/len(self.train_loader)
            print(f"main loss: {avg_main_loss:.6f} relation loss: {avg_relation_loss:.6f}")

            if epoch % metric_every == 0:
                self.validate(val_size, epoch, savename, avg_main_loss, avg_relation_loss)

    def validate(self, val_size, epoch, savename, avg_main_loss, avg_relation_loss):
        with torch.no_grad():
            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.teacher_generator(noised)
                n += 1
                if n == 1000:
                    break
            teacher_time = (time.time() - start) / 1000

            n = 0
            start = time.time()
            for (_, noised, _) in self.val_loader_unit_batch:
                _ = self.student_generator(noised)
                n += 1
                if n == 1000:
                    break
            student_time = (time.time() - start) / 1000

            teacher_ms = teacher_time * 1000
            student_ms = student_time * 1000

            print(f"avg teacher: {teacher_ms:.3f}ms, avg student: {student_ms:.3f}ms")

            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            step = 0
            for (real, noised, w) in self.val_loader:
                step += 1
                if step > val_size:
                    break
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

        info = {
            "teacher time, ms": teacher_ms
            , "student time, ms": student_ms
            , "main loss": avg_main_loss
            , "relation loss": avg_relation_loss
            , "epoch": epoch
        }

        plot_metrics(
            torch.cat(real_batches, dim=0)
            , torch.cat(teacher_gen_batches, dim=0)
            , torch.cat(student_gen_batches, dim=0)
            , torch.cat(w_batches, dim=0)
            , info
            , savename
        )


class StudentTrainerExp:
    def __init__(
        self
        , train_loader
        , val_loader
        , val_loader_unit_batch
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , start_lr
        , teacher_generator
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_unit_batch = val_loader_unit_batch

        self.student_num_layers = student_num_layers

        self.teacher_generator = teacher_generator
        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
            , 128
        ).to(device)

        self.start_lr = start_lr

        print(self.teacher_generator)
        print(self.student_generator)

        self.epochs = epochs

        self.rvs_history = []
        self.tvs_history = []
        self.mse_history = []
        self.epoch_history = []

    def train(self, metric_every, savename, val_size=None):
        if not val_size:
            val_size = len(self.val_loader)

        self.optimizer = AdamW(self.student_generator.parameters(), lr=self.start_lr)
        lr_track = np.logspace(0, -0.01, num=self.epochs)
        lr_lambda = lambda x: lr_track[x]
        self.sheduler = LambdaLR(self.optimizer, lr_lambda)
        for epoch in range(1, self.epochs+1):
            print(f"epoch {epoch} (idle)")

            avg_loss = 0

            for (real, noised, w) in tqdm(self.train_loader):
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                loss = ((teacher_generated - student_generated) ** 2).mean()

                avg_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch != self.epochs:
                self.sheduler.step()
            avg_loss = avg_loss/len(self.train_loader)
            print(f"train loss: {avg_loss:.6f}")

            if epoch % metric_every == 0:
                teacher_vs_student = self.validate(val_size, epoch, savename)

                self.tvs_history.append(teacher_vs_student)
                self.mse_history.append(avg_loss)
                self.epoch_history.append(epoch)

                plt.plot(self.epoch_history, self.mse_history, marker='o')
                plt.tight_layout()
                plt.savefig(f"exp/mse_{epoch}e.png", dpi=300)
                plt.show()

                plt.plot(self.epoch_history, self.tvs_history, marker='o')
                plt.tight_layout()
                plt.savefig(f"exp/tvs_{epoch}e.png", dpi=300)
                plt.show()



    def validate(self, val_size, epoch, savename):
        with torch.no_grad():
            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            step = 0
            for (real, noised, w) in self.val_loader:
                step += 1
                if step > val_size:
                    break
                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

        info = {
            "epoch": epoch
        }

        return plot_metrics_exp(
            torch.cat(real_batches, dim=0)
            , torch.cat(teacher_gen_batches, dim=0)
            , torch.cat(student_gen_batches, dim=0)
            , torch.cat(w_batches, dim=0)
            , info
            , savename
        )
