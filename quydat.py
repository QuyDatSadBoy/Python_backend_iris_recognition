import os

def process_yolo_dataset(dataset_path=r'/home/quydat09/iris_rcog/eye-recognition-service/datasets/quydat'):

    
    # Tìm tất cả các ảnh và file label trong thư mục
    eye_data_list = []
    image_path_file = []
    label_path_file = []
    
    # Tìm tất cả các thư mục trong dataset
    for root, dirs, files in os.walk(dataset_path):
        # Kiểm tra mỗi file trong thư mục
        image_path_file += [{ 'path':os.path.join(root, f.strip()),'name':f.strip()[:-4]} for f in files if f.endswith('.jpg') or f.endswith('.png')]
        label_path_file += [{'path':os.path.join(root, f.strip()),'name':f.strip()[:-4]} for f in files if f.endswith('.xml') or f.endswith('.txt')]
        
    # print(image_path_file[0:3])
    # print(label_path_file[0:3])
    # exit(0)
    
    # print(image_path_file)
    # print(label_path_file)
    # print(len(image_path_file))
    # print(len(label_path_file))
    cnt = 0
    
    for img_path in image_path_file:
        for label_path in label_path_file:
          if img_path['name'] == label_path['name']:
            # print(img_path['path'],label_path['path'])
            cnt += 1
    print(cnt)

process_yolo_dataset()