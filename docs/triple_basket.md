# Task riêng: "put all three items in the basket"

Task LIBERO mới, tích hợp qua plugin `plugins/lerobot_env_triple_basket` (không sửa `lerobot/`).
Trạng thái: **scene, tiêu chí thành công và script đã viết nhưng CHƯA được chạy trên simulator** (môi trường soạn thảo không có GPU/.venv). Mọi bước dưới đây cần chạy và chỉnh trên máy có cài đặt.

## 1. Đặc tả
- Instruction: "put all three items in the basket"; suite `libero_triple_basket`, task id 0.
- `item_1`, `item_2`, `item_3`: 3 chai ketchup giống hệt (LIBERO yêu cầu mỗi vật thuộc một category có sẵn), mỗi vật một ID riêng.
- Vị trí cố định trên sàn (frame LIBERO floor): item_1 (-0.17,-0.18), item_2 (0.00,-0.18), item_3 (0.17,-0.18); cách nhau 0.17 m. Rổ `basket_1` tại (0.0, 0.26). Dung sai ±2 mm, nên mỗi lần reset gần như trùng nhau.
- Định nghĩa tại `.../data/bddl/put_all_three_items_in_the_basket.bddl`.

## 2. Tích hợp (không sửa vendor)
LeRobot tự import mọi package tên `lerobot_env_*` (`register_third_party_plugins`). Plugin đăng ký `LIBERO_TRIPLE_BASKET` vào `libero.benchmark`; `problem_folder` là đường dẫn tuyệt đối nên LeRobot lấy BDDL/init-state từ package.
```bash
.venv/bin/pip install -e plugins/lerobot_env_triple_basket
.venv/bin/python scripts/triple_basket/generate_init_states.py   # tạo trạng thái đầu cố định
```

## 3. Thành công
BDDL goal = `(And (In item_1 basket_1_contain_region) (In item_2 ...) (In item_3 ...))`: kiểm tra từng vật nằm trong vùng chứa của rổ (contact + trong box), AND cả ba. Plugin thêm điều kiện giữ yên `SUCCESS_DWELL_STEPS = 5` bước liên tiếp rồi mới trả `is_success=True` (LeRobot ghi vào `eval_info.json`).

## 4. Kiểm tra scene
`.venv/bin/python scripts/triple_basket/verify_scene.py` kiểm tra: reset lặp lại, khoảng cách, tầm với, từng vật thả vào rổ được nhận diện, chỉ báo thành công khi đủ cả ba, và hủy khi một vật rời rổ. `scripted_expert.py` là controller dùng trạng thái đặc quyền; hệ số/độ cao cần tinh chỉnh (ketchup nằm ngang, hướng gripper có thể cần thêm xoay).

## 5. Dữ liệu và fine-tune
`collect_demos.py [N]` lưu các lượt thành công (ảnh agentview + wrist, state, action 7-D) vào `outputs/triple_basket_demos/*.npz` (ngoài Git). Bước còn thiếu: chuyển sang `LeRobotDataset` rồi fine-tune từ `HuggingFaceVLA/smolvla_libero` bằng `lerobot-train` (cùng schema ảnh/state như dữ liệu LIBERO). Checkpoint hiện tại chưa được dạy chuỗi 3 lần gắp nên không nên kỳ vọng nó tự làm được.

## 6. Đánh giá
```bash
bash scripts/run_triple_basket.sh <checkpoint_đã_fine-tune> outputs/triple_run 10
```
Xem `per_task[0].metrics.successes` trong `eval_info.json` và video `videos/libero_triple_basket_0/` để xác nhận cả ba vật thật sự ở trong rổ. `episode_length=900` vì 3 lần gắp dài hơn mặc định 500 bước của suite không có trong bảng của LeRobot.
