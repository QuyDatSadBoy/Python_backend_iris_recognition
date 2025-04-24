from db_connection import get_connection
from entity.EyeRecognitionSample import DetectEyeDataTrain, DetectEyeData
from typing import List, Optional, Dict, Any
import os


class EyeRecognitionSampleDAO:
    """
    Data Access Object for DetectEyeDataTrain and DetectEyeData
    """
    
    @staticmethod
    def get_all_data_trains() -> List[Dict[str, Any]]:
        """
        Get all data training sets from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeDataTrain"
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_data_train_by_id(data_train_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a data training set by its ID
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeDataTrain WHERE id = %s"
                cursor.execute(sql, (data_train_id,))
                result = cursor.fetchone()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def create_data_train(data_train: DetectEyeDataTrain) -> int:
        """
        Create a new data training set in the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblDetectEyeDataTrain 
                (dataTrainPath, detailFilePath) 
                VALUES (%s, %s)
                """
                cursor.execute(sql, (
                    data_train.dataTrainPath,
                    data_train.detailFilePath
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
    
    @staticmethod
    def update_data_train(data_train_id: int, data_train: DetectEyeDataTrain) -> bool:
        """
        Update an existing data training set
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblDetectEyeDataTrain 
                SET dataTrainPath = %s, detailFilePath = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    data_train.dataTrainPath,
                    data_train.detailFilePath,
                    data_train_id
                ))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
    
    @staticmethod
    def delete_data_train(data_train_id: int) -> bool:
        """
        Delete a data training set from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                # First delete all associated eye data
                sql = "DELETE FROM tblDetectEyeData WHERE tblDetectEyeDataTrainId = %s"
                cursor.execute(sql, (data_train_id,))
                
                # Then delete the data train
                sql = "DELETE FROM tblDetectEyeDataTrain WHERE id = %s"
                cursor.execute(sql, (data_train_id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
    
    @staticmethod
    def get_all_eye_data(data_train_id: int) -> List[Dict[str, Any]]:
        """
        Get all eye data for a specific data training set
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeData WHERE tblDetectEyeDataTrainId = %s"
                cursor.execute(sql, (data_train_id,))
                result = cursor.fetchall()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_eye_data_by_id(eye_data_id: int) -> Optional[Dict[str, Any]]:
        """
        Get eye data by its ID
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeData WHERE id = %s"
                cursor.execute(sql, (eye_data_id,))
                result = cursor.fetchone()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def create_eye_data(eye_data: DetectEyeData) -> int:
        """
        Create a new eye data entry in the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblDetectEyeData 
                (imageLink, labelLink, tblDetectEyeDataTrainId) 
                VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (
                    eye_data.imageLink,
                    eye_data.labelLink,
                    eye_data.tblDetectEyeDataTrainId
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
    
    @staticmethod
    def create_multiple_eye_data(eye_data_list: List[DetectEyeData]) -> int:
        """
        Create multiple eye data entries in the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblDetectEyeData 
                (imageLink, labelLink, tblDetectEyeDataTrainId) 
                VALUES (%s, %s, %s)
                """
                data = [(item.imageLink, item.labelLink, item.tblDetectEyeDataTrainId) for item in eye_data_list]
                cursor.executemany(sql, data)
                connection.commit()
                return cursor.rowcount
        finally:
            connection.close()
    
    @staticmethod
    def update_eye_data(eye_data_id: int, eye_data: DetectEyeData) -> bool:
        """
        Update an existing eye data entry
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblDetectEyeData 
                SET imageLink = %s, labelLink = %s, tblDetectEyeDataTrainId = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    eye_data.imageLink,
                    eye_data.labelLink,
                    eye_data.tblDetectEyeDataTrainId,
                    eye_data_id
                ))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
    
    @staticmethod
    def delete_eye_data(eye_data_id: int) -> bool:
        """
        Delete an eye data entry from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "DELETE FROM tblDetectEyeData WHERE id = %s"
                cursor.execute(sql, (eye_data_id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()