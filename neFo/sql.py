
import sqlite3
import numpy as np

feature_list = []
ids_list = []
image_name_list = []

con = sqlite3.connect('/home/anhp/Downloads/LLQ_f32.db')
cursor = con.cursor()
for folder_name, image_name, feature in cursor.execute('SELECT Name, Image_Path, Feature FROM Person_Infor AS A \
                                                        JOIN FaceDict AS B WHERE A.ID == B.Person_ID '):
    image_name_list.append(image_name)
    feature_list.append(np.fromstring(feature, dtype=float, sep=','))
    ids_list.append(folder_name)

feature_list = np.array(feature_list, dtype=np.float32)
print(f'Origin DB has {len(set(ids_list))} users')
print(len(image_name_list), len(feature_list), len(ids_list))
print('--------------------------------------')
con.close()