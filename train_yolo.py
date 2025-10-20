# from ultralytics import YOLO
# import os

# # ตรวจสอบว่าไฟล์โมเดลเริ่มต้น (yolo11n.pt) อยู่ในไดเรกทอรีเดียวกันกับสคริปต์นี้
# # หรือระบุ path ที่ถูกต้อง    
# # model_path = 'yolo11n.pt'
# model_path = r"runs/detect/banana_ripeness_detector/weights/last.pt"

# if not os.path.exists(model_path):
#     print(f"Error: Model file '{model_path}' not found.")
#     print("Please ensure 'yolo11n.pt' is in the same directory as 'train_yolo.py'")
#     exit()

# # โหลดโมเดล YOLO ที่มีอยู่ (yolo11n.pt)
# # นี่คือโมเดลที่ถูกฝึกมาแล้ว และเราจะนำมา Fine-tuning ต่อ
# model = YOLO(model_path)

# # กำหนด path ไปยังไฟล์ data.yaml ที่สร้าง
# # ให้แน่ใจว่า path นี้ถูกต้อง
# # เปลี่ยนชื่อไฟล์ YAML ให้ตรงกับชื่อ (จาก 'banana_classification.yaml' เป็น 'data.yaml')
# data_yaml_path = 'data.yaml' 
# if not os.path.exists(data_yaml_path):
#     print(f"Error: Data YAML file '{data_yaml_path}' not found.")
#     print("Please ensure 'data.yaml' is in the same directory as 'train_yolo.py'") # <--- แก้ไขข้อความแจ้งเตือน
#     exit()

# print(f"Starting fine-tuning with data: {data_yaml_path}")
# print(f"Using initial model: {model_path}")

# # เริ่มการฝึกโมเดล
# # data: Path ไปยังไฟล์ data.yaml
# # epochs: จำนวนรอบการฝึก (สามารถปรับได้) - เริ่มต้นด้วยค่าที่น้อยก่อน เช่น 50-100
# # imgsz: ขนาดภาพที่โมเดลจะใช้ในการฝึก (ควรตรงกับที่คุณตั้งค่าใน Roboflow)
# #        640 เป็นขนาดมาตรฐานที่ดี
# # batch: ขนาดของ batch (จำนวนภาพที่ใช้ในการคำนวณในแต่ละครั้ง) - ปรับตาม RAM ของ GPU
# #        ถ้า RAM น้อย ให้ลดค่า batch ลง เช่น 8, 4
# # name: ชื่อของ run การฝึกนี้ ผลลัพธ์จะถูกบันทึกใน runs/detect/your_run_name
# results = model.train(data=data_yaml_path, epochs=100, imgsz=640, batch=16, name='banana_ripeness_detector')

# print("\nFine-tuning completed!")
# print("Trained model weights are typically saved in: runs/detect/banana_ripeness_detector/weights/")
# print("Look for 'best.pt' or 'last.pt' in that directory.")

# # คุณสามารถใช้โมเดลที่ฝึกเสร็จแล้วเพื่อทดสอบได้ทันที
# # model_trained = YOLO('runs/detect/banana_ripeness_detector/weights/best.pt')
# # results_predict = model_trained('path/to/your/test_image.jpg')
# # results_predict[0].show() # แสดงผลลัพธ์


# from ultralytics import YOLO
# import os

# # ใช้ checkpoint ล่าสุดแทน yolo11n.pt
# model_path = r"C:\projexts\banana_classification\runs\detect\banana_ripeness_detector23\weights\last.pt"

# if not os.path.exists(model_path):
#     print(f"Error: Model file '{model_path}' not found.")
#     print("Please ensure the path is correct")
#     exit()

# # โหลดโมเดลจาก last.pt 
# model = YOLO(model_path)

# # path ของ dataset yaml
# data_yaml_path = 'data.yaml'
# if not os.path.exists(data_yaml_path):
#     print(f"Error: Data YAML file '{data_yaml_path}' not found.")
#     exit()

# print(f"Resuming training with data: {data_yaml_path}")
# print(f"Using checkpoint: {model_path}")


# results = model.train(
#     data=data_yaml_path,
#     epochs=90,
#     imgsz=640,
#     batch=16,
#     name='banana_ripeness_detector23',  # ใช้ชื่อเดิม
#     resume=True
# )


# print("\nResume completed!")
# print("Updated weights saved in: runs/detect/banana_ripeness_detector/weights/")



# #--------#

from ultralytics import YOLO
import os

# ตรวจสอบว่าไฟล์โมเดลเริ่มต้น (yolo11n.pt) อยู่ในไดเรกทอรีเดียวกันกับสคริปต์นี้
# หรือระบุ path ที่ถูกต้อง
model_path = 'yolo11n.pt'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    print("Please ensure 'yolo11n.pt' is in the same directory as 'train_yolo.py'")
    exit()

# โหลดโมเดล YOLO ที่มีอยู่ (yolo11n.pt)
# นี่คือโมเดลที่ถูกฝึกมาแล้ว และเราจะนำมา Fine-tuning ต่อ
model = YOLO(model_path)

# กำหนด path ไปยังไฟล์ data.yaml ที่สร้าง
# ให้แน่ใจว่า path นี้ถูกต้อง
# เปลี่ยนชื่อไฟล์ YAML ให้ตรงกับชื่อ (จาก 'banana_classification.yaml' เป็น 'data.yaml')
data_yaml_path = 'data.yaml' 
if not os.path.exists(data_yaml_path):
    print(f"Error: Data YAML file '{data_yaml_path}' not found.")
    print("Please ensure 'data.yaml' is in the same directory as 'train_yolo.py'") # <--- แก้ไขข้อความแจ้งเตือน
    exit()

print(f"Starting fine-tuning with data: {data_yaml_path}")
print(f"Using initial model: {model_path}")

# เริ่มการฝึกโมเดล
# data: Path ไปยังไฟล์ data.yaml
# epochs: จำนวนรอบการฝึก (สามารถปรับได้) - เริ่มต้นด้วยค่าที่น้อยก่อน เช่น 50-100
# imgsz: ขนาดภาพที่โมเดลจะใช้ในการฝึก (ควรตรงกับที่คุณตั้งค่าใน Roboflow)
#        640 เป็นขนาดมาตรฐานที่ดี
# batch: ขนาดของ batch (จำนวนภาพที่ใช้ในการคำนวณในแต่ละครั้ง) - ปรับตาม RAM ของ GPU
#        ถ้า RAM น้อย ให้ลดค่า batch ลง เช่น 8, 4
# name: ชื่อของ run การฝึกนี้ ผลลัพธ์จะถูกบันทึกใน runs/detect/your_run_name
results = model.train(data=data_yaml_path, epochs=90, imgsz=640, batch=16, workers=0,
    cache='ram', name='banana_ripeness_detector')

print("\nFine-tuning completed!")
print("Trained model weights are typically saved in: runs/detect/banana_ripeness_detector/weights/")
print("Look for 'best.pt' or 'last.pt' in that directory.")


# # คุณสามารถใช้โมเดลที่ฝึกเสร็จแล้วเพื่อทดสอบได้ทันที
# # model_trained = YOLO('runs/detect/banana_ripeness_detector/weights/best.pt')
# # results_predict = model_trained('path/to/your/test_image.jpg')
# # results_predict[0].show() # แสดงผลลัพธ์

# #-------#
