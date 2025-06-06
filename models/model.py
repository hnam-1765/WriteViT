import sys
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CTCLoss
import os
import cv2
from params import *
from .BigGAN_networks import Discriminator
from util.util import (
    loss_hinge_dis,
    loss_hinge_gen,
    padding,
)

from data.dataset import TextDataset, TextDatasetval
import shutil
from .OCR import ViT_OCR
from .Generator import Generator
from .Writer import Writer, strLabelConverter
 

class WriteViT(nn.Module):

    def __init__(self, batch_size=batch_size):
        super(WriteViT, self).__init__()

        self.batch_size = batch_size
        self.epsilon = 1e-7
        self.netG = Generator().to(DEVICE)
        self.netD = nn.DataParallel(Discriminator()).to(DEVICE)
        self.netW =  Writer().to(DEVICE)
        self.netconverter = strLabelConverter(ALPHABET)
        self.netOCR = ViT_OCR().to(DEVICE)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction="none")

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=G_LR,
            betas=(0.0, 0.999),
            weight_decay=0,
            eps=1e-8,
        )
        self.optimizer_OCR = torch.optim.Adam(
            self.netOCR.parameters(),
            lr=OCR_LR,
            betas=(0.0, 0.999),
            weight_decay=0,
            eps=1e-8,
        )

        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=D_LR,
            betas=(0.0, 0.999),
            weight_decay=0,
            eps=1e-8,
        )

        self.optimizer_wl = torch.optim.Adam(
            self.netW.parameters(),
            lr=W_LR,
            betas=(0.0, 0.999),
            weight_decay=0,
            eps=1e-8,
        )
        self.optimizers = [
            self.optimizer_G,
            self.optimizer_OCR,
            self.optimizer_D,
            self.optimizer_wl,
        ]

        self.optimizer_G.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle1 = 0
        self.Lcycle2 = 0
        self.lda1 = 0
        self.lda2 = 0
        self.KLD = 0
        self.loss_patch_real = 0
        self.loss_patch_fake = 0
        self.loss_patch = 0

        with open(WORDS_PATH, "rb") as f:
            self.lex = f.read().splitlines()

        lex = []
        lex_upper_number = []

        for word in self.lex:
            try:
                word = word.decode("utf-8")
            except:
                continue

            if len(word) < 20:
                if word.isupper() or word.isdigit():
                    lex_upper_number.append(word)
                else:
                    lex.append(word)

        self.lex = lex
        self.lex_upper_number = lex_upper_number

        self.fake_y_dist = torch.distributions.Categorical(
            torch.tensor([1.0 / len(self.lex)] * len(self.lex))
        )
        my_string = MY_STRING
        self.text = [j.encode() for j in my_string.split(" ")]
        self.eval_text_encode, self.eval_len_text = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(DEVICE).repeat(
            self.batch_size, 1, 1
        )

    def save_images_for_fid_calculation(self, dataloader, epoch, mode="train"):

        self.real_base = os.path.join("saved_images", EXP_NAME, "Real")
        self.fake_base = os.path.join("saved_images", EXP_NAME, "Fake")

        if os.path.isdir(self.real_base):
            shutil.rmtree(self.real_base)
        if os.path.isdir(self.fake_base):
            shutil.rmtree(self.fake_base)

        os.mkdir(self.real_base)
        os.mkdir(self.fake_base)

        with torch.no_grad():
            index = 0
            for step, data in enumerate(dataloader):

                self.sdata = data["img"].to(DEVICE)
                self.words = [
                    word.encode("utf-8")
                    for word in np.random.choice(self.lex, self.batch_size)
                ]
                self.text_encode_fake, self.len_text_fake = self.netconverter.encode(
                    self.words
                )
                self.text_encode_fake = self.text_encode_fake.to(DEVICE)
                feat_w, _ = self.netW(
                    self.sdata.detach(), data["wcl"].to(DEVICE)
                )
                self.fakes = self.netG(feat_w, self.text_encode_fake)
                fake_images = self.fakes.detach().cpu().numpy()

                for i in range(fake_images.shape[0]):
                    for j in range(fake_images.shape[1]):
                        img = 255 * (((fake_images[i, j]) + 1) / 2)
                        img = padding(img)
                        cv2.imwrite(
                            os.path.join(
                                self.fake_base,
                                str(step * self.batch_size + i) + ".png",
                            ),
                            img,
                        )
                        index += 1

        if mode == "train":

            TextDatasetObj = TextDataset(num_examples=self.eval_text_encode.shape[1])
            dataset_real = torch.utils.data.DataLoader(
                TextDatasetObj,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                collate_fn=TextDatasetObj.collate_fn,
            )

        elif mode == "test":

            TextDatasetObjval = TextDatasetval(
                num_examples=self.eval_text_encode.shape[1]
            )
            dataset_real = torch.utils.data.DataLoader(
                TextDatasetObjval,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                collate_fn=TextDatasetObjval.collate_fn,
            )
        index = 0
        for step, data in enumerate(dataset_real):

            real_images = data["img"].numpy()

            for i in range(real_images.shape[0]):
                for j in range(real_images.shape[1]):
                    img = 255 * ((real_images[i, j] + 1) / 2)
                    img = padding(img)
                    cv2.imwrite(
                        os.path.join(
                            self.real_base,
                            str(step * self.batch_size + i) + ".png",
                        ),
                        img,
                    )

        return self.real_base, self.fake_base

    def _generate_page(
        self, img, ST, wcl,SLEN, eval_text_encode=None, eval_len_text=None
    ):

        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text

        feat_w = self.netW(img.detach(), wcl,training=False)
        
        self.fakes = self.netG.Eval(feat_w, eval_text_encode)

        page1s = []
        page2s = []

        for batch_idx in range(self.batch_size):

            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            for idx, fake_ in enumerate(self.fakes):

                word_t.append(
                    (
                        fake_[batch_idx, 0, :, : eval_len_text[idx] * resolution]
                        .cpu()
                        .numpy()
                        + 1
                    )
                    / 2
                )

                word_t.append(gap)

                if len(word_t) == 16 or idx == len(self.fakes) - 1:

                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page1 = np.concatenate(page_, 0)

            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            sdata_ = [i.unsqueeze(1) for i in torch.unbind(ST, 1)]

            for idx, st in enumerate((sdata_)):

                word_t.append(
                    (
                        st[batch_idx, 0, :, : int(SLEN.cpu().numpy()[batch_idx][idx])]
                        .cpu()
                        .numpy()
                        + 1
                    )
                    / 2
                )

                word_t.append(gap)

                if len(word_t) == 16 or idx == len(sdata_) - 1:

                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page2 = np.concatenate(page_, 0)

            merge_w_size = max(page1.shape[0], page2.shape[0])

            if page1.shape[0] != merge_w_size:

                page1 = np.concatenate(
                    [page1, np.ones([merge_w_size - page1.shape[0], page1.shape[1]])], 0
                )

            if page2.shape[0] != merge_w_size:

                page2 = np.concatenate(
                    [page2, np.ones([merge_w_size - page2.shape[0], page2.shape[1]])], 0
                )

            page1s.append(page1)
            page2s.append(page2)

            # page = np.concatenate([page2, page1], 1)

        page1s_ = np.concatenate(page1s, 0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = []

        for para in page2s:
            padded_page2s.append(
                np.concatenate(
                    [para, np.ones([para.shape[0], max_wid - para.shape[1]])], 1
                )
            )

        padded_page2s_ = np.concatenate(padded_page2s, 0)

        return np.concatenate([padded_page2s_, page1s_], 1)

    def get_current_losses(self):

        losses = {}

        losses["G"] = self.loss_G
        losses["D"] = self.loss_D
        losses["Dfake"] = self.loss_Dfake
        losses["Dreal"] = self.loss_Dreal
        losses["OCR_fake"] = self.loss_OCR_fake
        losses["OCR_real"] = self.loss_OCR_real
        losses["w_fake"] = self.loss_w_fake
        losses["w_real"] = self.loss_w_real
        losses["cycle1"] = self.Lcycle1
        losses["cycle2"] = self.Lcycle2
        losses["lda1"] = self.lda1
        losses["lda2"] = self.lda2
        losses["KLD"] = self.KLD
        losses["patch_real"] = self.loss_patch_real
        losses["patch_fake"] = self.loss_patch_fake
        losses["patch"] = self.loss_patch

        return losses

    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.opt.single_writer:
            load_filename = "%s_z.pkl" % (epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            self.z = torch.load(load_path)

    def _set_input(self, input):
        self.input = input

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):

        self.real = self.input["img"].to(DEVICE)
        self.label = self.input["label"]
        self.sdata = self.input["img"].to(DEVICE)
        self.ST_LEN = self.input["swids"]
        self.text_encode, self.len_text = self.netconverter.encode(self.label)

        self.text_encode = self.text_encode.to(DEVICE).detach()
        self.len_text = self.len_text.detach()

        sample_lex_idx = self.fake_y_dist.sample([self.batch_size])
        fake_y = [self.lex[i].encode("utf-8") for i in sample_lex_idx]
        
        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(fake_y)
        self.text_encode_fake = self.text_encode_fake.to(DEVICE)

    def backward_D_OCR_W(self):
        feat_w, self.loss_w_real = self.netW(
            self.real.detach(), self.input["wcl"].to(DEVICE)
        )
        _, self.pred_real_OCR = self.netOCR(self.real.detach())
        self.loss_w_real = self.loss_w_real.mean()
        self.fake = self.netG(feat_w, self.text_encode_fake)
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{"x": self.fake.detach()})


        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(
            pred_fake,
            pred_real,
            self.len_text_fake.detach(),
            self.len_text.detach(),
            True,
        )
        

        self.loss_D = self.loss_Dreal + self.loss_Dfake
        self.pred_real_OCR = self.pred_real_OCR.float()
        preds_size = torch.IntTensor(
            [self.pred_real_OCR.size(1)] * self.batch_size
        ).detach()
        self.pred_real_OCR = self.pred_real_OCR.permute(1, 0, 2).log_softmax(2)
        loss_OCR_real = self.OCR_criterion(
            self.pred_real_OCR,
            self.text_encode.detach(),
            preds_size,
            self.len_text.detach(),
        )
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        loss_total = (
            self.loss_D * 2 + self.loss_OCR_real + self.loss_w_real
        )
        # backward
        loss_total.backward()
        return loss_total

    def backward_G_only(self):

        self.gb_alpha = 0.7
        self.gb_beta = 0.7
        feat_w, _ = self.netW(self.real.detach(), self.input["wcl"].to(DEVICE))
        self.fake = self.netG(feat_w, self.text_encode_fake)
        pred_fake = self.netD(**{"x": self.fake})
        self.loss_G = loss_hinge_gen(
            pred_fake, self.len_text_fake.detach(), True
        ).mean()

        _, pred_fake_OCR = self.netOCR(self.fake)
        pred_fake_OCR = pred_fake_OCR.float()
        preds_size = torch.IntTensor([pred_fake_OCR.size(1)] * self.batch_size).detach()
        pred_fake_OCR = pred_fake_OCR.permute(1, 0, 2).log_softmax(2)
        loss_OCR_fake = self.OCR_criterion(
            pred_fake_OCR,
            self.text_encode_fake.detach(),
            preds_size,
            self.len_text_fake.detach(),
        )
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        _, self.loss_w_fake = self.netW(self.fake, self.input["wcl"].to(DEVICE))
        self.loss_w_fake = self.loss_w_fake.mean()
        self.loss_G = self.loss_G

        self.loss_T = (
            self.loss_G + self.loss_OCR_fake + self.loss_w_fake
        )

        grad_fake_OCR = torch.autograd.grad(
            self.loss_OCR_fake, self.fake, retain_graph=True
        )[0]
        grad_fake_WL = torch.autograd.grad(
            self.loss_w_fake, self.fake, retain_graph=True
        )[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, retain_graph=True)[
            0
        ]

        self.loss_T.backward(retain_graph=True)

        if True:
            grad_fake_OCR = torch.autograd.grad(
                self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True
            )[0]
            grad_fake_adv = torch.autograd.grad(
                self.loss_G, self.fake, create_graph=True, retain_graph=True
            )[0]
            grad_fake_WL = torch.autograd.grad(
                self.loss_w_fake, self.fake, create_graph=True, retain_graph=True
            )[0]
            gp_ocr = self.gb_alpha * torch.div(
                torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR)
            )
            gp_wl = self.gb_beta * torch.div(
                torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_WL)
            )
            self.loss_OCR_fake = gp_ocr.detach() * self.loss_OCR_fake
            self.loss_w_fake = gp_wl.detach() * self.loss_w_fake
            self.loss_T = (
                self.loss_G * 2 + self.loss_OCR_fake + self.loss_w_fake
            )
            self.loss_T.backward(retain_graph=True)
            with torch.no_grad():
                self.loss_T.backward()

    def optimize_D_OCR_W(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.set_requires_grad([self.netW], True)
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_wl.zero_grad()
        self.backward_D_OCR_W()

    def optimize_D_OCR_W_step(self):

        self.optimizer_D.step()
        self.optimizer_wl.step()
        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_wl.zero_grad()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_only()

    def optimize_G_step(self):

        self.optimizer_G.step()
        self.optimizer_G.zero_grad()
