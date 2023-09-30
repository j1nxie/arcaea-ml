import json
import os
import time

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
    https://zhuanlan.zhihu.com/p/28527749
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.


    """  # noqa: E501

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, 1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print("probs size= {}".format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print("-----batch_loss------")
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def dump_index_songid_relation(base_dir):
    dump_data = {"idx2id": [], "id2idx": {}, "time": int(time.time() * 1000)}
    with open(f"{base_dir}/songlist") as f:
        songlist = json.load(f)
        songs = songlist["songs"]

    index = 0
    for song in songs:
        dump_data["idx2id"].append(song["id"])
        dump_data["id2idx"][song["id"]] = index
        index += 1
    with open("index_songid_relation.json", "w", encoding="utf-8") as f:
        json.dump(dump_data, f, ensure_ascii=False)
    with open("gen_time.txt", "w") as f:
        f.write(str(dump_data["time"]))
    return dump_data["idx2id"], dump_data["id2idx"]


def load_images(device, base_dir):
    img_map = {}
    item_id_map = {}
    img_files = []
    empty_collect = []
    weights = []

    with open(f"{base_dir}/songlist") as f:
        songlist = json.load(f)
        songs = songlist["songs"]

    for song in songs:
        folder_name = song["id"]
        if song.get("remote_dl"):
            folder_name = f"dl_{folder_name}"

        weight = 0

        for filename in os.listdir(f"{base_dir}/{folder_name}"):
            # if not filename.endswith(".jpg"):
            #   continue
            if not filename.endswith("_256.jpg"):
                continue

            full_path = f"{base_dir}/{folder_name}/{filename}"

            img_files.append(full_path)
            item_id_map[full_path] = song["id"]
            weight += 1

            with open(full_path, "rb") as f:
                nparr = np.frombuffer(f.read(), np.uint8)
                # convert to image array
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (60, 60))
                img_map[full_path] = (
                    torch.from_numpy(np.transpose(image, (2, 0, 1))).float().to(device)
                )
        if weight == 0:
            print(f"empty_collect: {song['id']}")
            empty_collect.append(song["id"])
            continue
        weights.append(weight)

    weights_t = torch.as_tensor(weights)
    weights_t[weights_t > 160] = 160
    weights_t = 1 / weights_t
    return img_map, img_files, item_id_map, weights_t, empty_collect


def get_data(img_files, item_id_map, img_map, id2idx):
    images = []
    labels = []
    for filepath in img_files:
        item_id = item_id_map[filepath]

        for _ in range(4):
            image_aug = img_map[filepath]
            images.append(image_aug)
            labels.append(id2idx[item_id])

    images_t = torch.stack(images)
    labels_t = torch.from_numpy(np.array(labels)).long().to(device)

    return images_t, labels_t


class Cnn(nn.Module):
    def __init__(self, num_class: int):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=3, padding=2),  # 16 * 20 * 20
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.AvgPool2d(5, 5),  # 16 * 4 * 4
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 16 * 2 * 2
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.AvgPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 2 * num_class),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(2 * num_class, num_class),
        )

    def forward(self, x):
        x /= 255.0
        out = self.conv(x)
        out = out.reshape(-1, 16 * 2 * 2)
        out = self.fc(out)
        return out


def train(device, base_dir, img_map, img_files, item_id_map, weights_t, idx2id, id2idx):
    num_class = len(idx2id)
    criterion = FocalLoss(num_class, alpha=weights_t)
    criterion.to(device)

    def compute_loss(x, label):
        loss = criterion(x, label)
        prec = (x.argmax(1) == label).float().mean()
        return loss, prec

    print(f"Training on {device}")
    model = Cnn(num_class).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    step = 0
    prec = 0
    target_step = 1200
    last_time = time.monotonic()
    is_saved = False
    best = 999
    while step < target_step or not is_saved:
        images_t, labels_t = get_data(img_files, item_id_map, img_map, id2idx)
        optim.zero_grad()
        score = model(images_t)
        loss, prec = compute_loss(score, labels_t)
        loss.backward()
        optim.step()
        if step < 10 or step % 50 == 0:
            print(step, loss.item(), prec.item(), time.monotonic() - last_time)
            last_time = time.monotonic()
        step += 1
        if step > target_step - 300 and best > loss.item():
            model.eval()
            if test(base_dir, model, idx2id):
                best = loss.item()
                print(f"save best {best}")
                model.train()
                torch.save(model.state_dict(), "./model.pth")
                torch.onnx.export(
                    model, torch.rand((1, 3, 60, 60)).to(device), "arcaea_jackets.onnx"
                )
                is_saved = True
            else:
                model.train()
        if step > target_step * 2:
            raise Exception("train too long")
    add_metadata_to_model("arcaea_jackets.onnx")


def add_metadata_to_model(path):
    with open("index_songid_relation.json") as f:
        content = f.read()
    import onnx

    model = onnx.load_model(path)
    meta = model.metadata_props.add()
    meta.key = "relation"
    meta.value = content
    onnx.save_model(model, path)


def predict(model, roi_list, idx2id):
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float().to(device)
    with torch.no_grad():
        score = model(roi_t)
        probs = nn.Softmax(1)(score)
        predicts = score.argmax(1)

    probs = probs.cpu().data.numpy()
    predicts = predicts.cpu().data.numpy()
    return [(idx2id[idx], idx) for idx in predicts], [
        probs[i, predicts[i]] for i in range(len(roi_list))
    ]


def test(base_dir, model, idx2id):
    with open(f"{base_dir}/songlist") as f:
        songlist = json.load(f)
        songs = songlist["songs"]

    items = []
    expect_ids = []
    for song in songs:
        folder_name = song["id"]
        if song.get("remote_dl"):
            folder_name = f"dl_{folder_name}"

        for filename in os.listdir(f"{base_dir}/{folder_name}"):
            if not filename.endswith("_256.jpg"):
                continue

            full_path = f"{base_dir}/{folder_name}/{filename}"
            with open(full_path, "rb") as f:
                nparr = np.frombuffer(f.read(), np.uint8)
                # convert to image array
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (60, 60))
                items.append(image)
                expect_ids.append(song["id"])

    roi_list = []
    for x in items:
        roi = np.transpose(x, (2, 0, 1))
        roi_list.append(roi)

    predicted_items, probabilities = predict(model, roi_list, idx2id)

    for i in range(len(predicted_items)):
        item_id = predicted_items[i][0]
        expect_id = expect_ids[i]
        # print(f"{item_id}/{expect_id}, {probabilities[i]:.3f}")
        thresh = 0.7
        if item_id != expect_id:
            # cv2_imshow(items[i])
            # inventory.show_img(items[i])
            print(f"Wrong predict: {item_id}/{expect_id}, {probabilities[i]}")
            return False
        elif probabilities[i] < thresh:
            print(f"low confidence: {item_id}/{expect_id}, {probabilities[i]}")
            return False
    return True


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

img_map, img_files, item_id_map, weights_t, empty_collect = load_images(device, "songs")
idx2id, id2idx = dump_index_songid_relation("songs")
try:
    train(device, "songs", img_map, img_files, item_id_map, weights_t, idx2id, id2idx)
except Exception:
    # print(torch.cuda.memory_summary())
    raise
