# Hướng dẫn thu dữ liệu và train baseline cho task mới

Dành cho thành viên thu dữ liệu mô phỏng Panthera-HT trong RoboTwin. Mỗi người nhận 2 task. Với mỗi task, bạn giao lại:

1. Một dataset LeRobot trên Hugging Face Hub, gồm 200 demo sạch và 200 demo DART.
2. Một Diffusion Policy (DP) đã train, kèm kết quả eval 20 episode.

| Người | Task |
| --- | --- |
| A | `move_stapler_pad`, `place_mouse_pad` |
| B | `place_empty_cup`, `stack_blocks_two` |

Task `move_pillbottle_pad` đã làm xong. Đó là ví dụ mẫu: DP đạt 11/20 trên eval chuẩn, dataset ở [Kevildarknight/panthera_move_pillbottle_pad](https://huggingface.co/datasets/Kevildarknight/panthera_move_pillbottle_pad).

## 0. Máy cần có

- Ubuntu 24.04 (hoặc 22.04) chạy trực tiếp, không dùng WSL. Cần GPU NVIDIA từ 4 GB và driver hỗ trợ Vulkan.
- Ổ trống khoảng 40 GB cho môi trường và assets, cộng thêm khoảng 20 GB cho mỗi task.
- Tài khoản Hugging Face và một token **Write** của riêng bạn. Đăng nhập bằng `hf auth login`. Không gửi token cho ai, kể cả trong nhóm chat.

## 1. Cài đặt (một lần)

Làm theo mục 1–3 của [README.md](README.md) trong thư mục này. Các mục đó gồm:

- clone `6dofvla` (nhánh `robotwin-data`) và `Panthera-HT_RoboTwin`;
- tạo conda env `RoboTwin`;
- tải assets;
- chạy `bash scripts/install.sh`.

Hai lỗi hay gặp:

- **Chưa cài git-lfs trước khi clone.** File URDF của robot sẽ chỉ là file trỏ, và SAPIEN không nạp được robot. README có lệnh kiểm tra.
- **IPv6 hỏng.** Tải từ Hugging Face sẽ treo khoảng 40 s mỗi kết nối. Cách sửa có trong README (dùng `gai.conf` hoặc `wget -4`).

Kiểm tra lại cài đặt:

```bash
conda run -n RoboTwin python -c "import torch, sapien, curobo; print(torch.cuda.is_available())"
ls ~/6dofvla/.venv/bin/lerobot-train
```

## 2. Quy trình cho một task

Mọi lệnh chạy từ `~/6dofvla`. Đặt tên task một lần ở đầu:

```bash
cd ~/6dofvla
T=move_stapler_pad          # đổi thành task của bạn
```

GPU 4 GB không chạy được hai việc cùng lúc, nên làm **lần lượt từng bước**. Thời gian trong ngoặc là ước tính trên RTX 3050 Ti.

### 2.1 Chạy thử 2 episode (khoảng 10 phút)

```bash
bash scripts/robotwin/collect.sh $T ${T}_panthera_single_30hz 2
```

Mở video head camera trong `~/panthera/Panthera-HT_RoboTwin/RoboTwin/data/$T/${T}_panthera_single_30hz/video/` và kiểm tra:

- robot là tay Panthera một tay;
- vật được đặt đúng chỗ.

Nếu lỗi, gửi log cho nhóm trước khi chạy tiếp.

### 2.2 Thu 200 demo sạch (8–10 giờ)

```bash
bash scripts/robotwin/collect.sh $T ${T}_panthera_single_30hz > outputs/collect_${T}.log 2>&1
```

Lệnh này chạy tiếp từ 2 episode thử ở bước 2.1. Nếu phải dừng giữa chừng (Ctrl+C hoặc tắt máy), chạy lại đúng lệnh này và nó sẽ tiếp tục từ seed cuối.

RoboTwin chạy 2 giai đoạn:

1. Lên kế hoạch quỹ đạo cho đủ 200 episode, in `simulate data episode N success!`.
2. Render và lưu dữ liệu.

Dòng `failed N times / M tries` là bình thường: các seed lỗi được bỏ qua.

**Không sửa file `.yml` sau khi đã bắt đầu thu.** RoboTwin tính dấu vân tay (fingerprint) từ file này. Nếu file thay đổi, RoboTwin sẽ từ chối chạy tiếp vào cùng thư mục. Muốn đổi số episode, thêm số đó vào cuối lệnh (như `2` ở bước 2.1), đừng sửa file.

### 2.3 Thu 200 demo DART (khoảng 6 giờ)

```bash
DART=1 bash scripts/robotwin/collect.sh $T ${T}_panthera_single_30hz_dart > outputs/collect_${T}_dart.log 2>&1
```

DART đẩy tay robot lệch khỏi quỹ đạo chuẩn, rồi ghi lại cách chuyên gia sửa về. Với task lọ thuốc, dữ liệu này giúp policy bớt làm đổ vật khi gắp.

### 2.4 Chuyển sang LeRobot và gộp (khoảng 1 giờ)

```bash
R=~/panthera/Panthera-HT_RoboTwin/RoboTwin/data/$T
bash scripts/robotwin/convert.sh $R/${T}_panthera_single_30hz
bash scripts/robotwin/convert.sh $R/${T}_panthera_single_30hz_dart

N=robotwin_${T}_panthera_single_30hz_clean_dart
.venv/bin/lerobot-edit-dataset --new_repo_id=local/$N --new_root=data/$N --operation.type=merge \
  --operation.repo_ids="[\"local/robotwin_${T}_panthera_single_30hz\",\"local/robotwin_${T}_panthera_single_30hz_dart\"]" \
  --operation.roots="[\"data/robotwin_${T}_panthera_single_30hz\",\"data/robotwin_${T}_panthera_single_30hz_dart\"]"
```

Kiểm tra kết quả: số frame trong `data/$N/meta/info.json` phải bằng tổng số frame của hai dataset nguồn.

Log có thể in traceback `libavutil.so.56` hoặc `torchcodec`. Lỗi này vô hại: LeRobot tự chuyển sang dùng pyav.

### 2.5 Đẩy lên Hub

```bash
.venv/bin/python scripts/robotwin/push_dataset.py data/$N <hf_user>/panthera_$T --public
```

Script tự nhận task từ tên thư mục và tạo dataset card.

### 2.6 Train Diffusion Policy (khoảng 9 giờ)

```bash
DATASET=$N STEPS=60000 bash scripts/robotwin/train_diffusion.sh outputs/dp_$T > outputs/train_dp_$T.log 2>&1
```

Checkpoint được lưu ở bước 20k, 40k và 60k, mỗi cái khoảng 1 GB. Muốn tạm dừng thì Ctrl+C. Để train tiếp, dùng lệnh dưới đây. **Phải có biến `PYTORCH_CUDA_ALLOC_CONF`**, nếu không sẽ bị lỗi hết bộ nhớ GPU:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/robotwin/train_diffusion.py \
  --config_path=outputs/dp_$T/checkpoints/last/pretrained_model/train_config.json \
  --resume=true >> outputs/train_dp_$T.log 2>&1
```

### 2.7 Eval (khoảng 1 giờ mỗi checkpoint)

```bash
for c in 040000 060000; do
  TASK_NAME=$T TASK_CONFIG=${T}_panthera_single_30hz \
    bash scripts/robotwin/eval.sh outputs/dp_$T/checkpoints/$c/pretrained_model 20 > outputs/eval_dp_${T}_$c.log 2>&1
done
```

Kết quả nằm trong `~/panthera/Panthera-HT_RoboTwin/RoboTwin/eval_result/$T/smolvla_panthera/${T}_panthera_single_30hz/dp_${T}_<checkpoint>/<thời gian>/`. Thư mục này gồm:

- `_result.txt`: tỉ lệ thành công;
- `episodeN.mp4`: video từng episode.

Eval dùng 20 seed bắt đầu từ 100000 với texture chưa thấy khi train, giống eval của task lọ thuốc, nên kết quả so sánh trực tiếp được.

## 3. Giao lại cho nhóm

Với mỗi task, gửi:

1. Link dataset trên Hub.
2. Kết quả eval theo mẫu:

   | Task | Dữ liệu (episode / frame) | DP 040000 | DP 060000 |
   | --- | --- | --- | --- |
   | move_pillbottle_pad | 885 / 108,426 | … | … |

3. 2–3 video thất bại tiêu biểu, kèm một câu mô tả lỗi. Ví dụ: "gắp hụt", "làm đổ vật", "mang tới nơi nhưng không thả".

## 4. Lưu ý

- Không chạy thu dữ liệu, train và eval cùng lúc trên GPU 4 GB.
- Không sửa file `.sh` khi nó đang chạy. Bash đọc script dần trong lúc chạy, nên chỉnh giữa chừng sẽ làm hỏng lần chạy đó.
- Không commit `data/` hay `outputs/`. `.gitignore` của repo chỉ cho phép commit script và cấu hình.
- Khi ổ đầy: sau khi đã đẩy dataset gộp lên Hub, có thể xoá dữ liệu RoboTwin gốc (`RoboTwin/data/$T/`, khoảng 10 GB). Chỉ nên xoá khi chắc không cần thu thêm vào thư mục đó.
