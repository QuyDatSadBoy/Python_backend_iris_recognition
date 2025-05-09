from db_connection import get_connection
from entity.DetectEyeDataTrain import DetectEyeDataTrain
from typing import List, Optional, Dict

class DetectEyeDataTrainDAO:
    
    @staticmethod
    def getAllDetectEyeDataTrain() -> List[DetectEyeDataTrain]:
        """Lấy tất cả training data"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeDataTrain"
                cursor.execute(sql)
                results = cursor.fetchall()
                return [DetectEyeDataTrain(**result) for result in results]
        finally:
            connection.close()
    
    @staticmethod
    def getDetectEyeDataTrainById(id: int) -> Optional[DetectEyeDataTrain]:
        """Lấy training data theo id"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeDataTrain WHERE id = %s"
                cursor.execute(sql, (id,))
                result = cursor.fetchone()
                return DetectEyeDataTrain(**result) if result else None
        finally:
            connection.close()
            
    @staticmethod
    def createDetectEyeDataTrain(detectEyeDataTrain: DetectEyeDataTrain) -> int:
        """Tạo training data mới"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblDetectEyeDataTrain 
                (dataTrainPath, detailFilePath) 
                VALUES (%s, %s)
                """
                cursor.execute(sql, (
                    detectEyeDataTrain.dataTrainPath,
                    detectEyeDataTrain.detailFilePath
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
            
    @staticmethod
    def updateDetectEyeDataTrain(id: int, detectEyeDataTrain: DetectEyeDataTrain) -> bool:
        """Cập nhật training data"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblDetectEyeDataTrain 
                SET dataTrainPath = %s, detailFilePath = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    detectEyeDataTrain.dataTrainPath,
                    detectEyeDataTrain.detailFilePath,
                    id
                ))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
            
    @staticmethod
    def deleteDetectEyeDataTrain(id: int) -> bool:
        """Xóa training data"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                # Xóa dữ liệu liên quan trước
                sql = "DELETE FROM tblDetectEyeData WHERE tblDetectEyeDataTrainId = %s"
                cursor.execute(sql, (id,))
                
                # Xóa training data
                sql = "DELETE FROM tblDetectEyeDataTrain WHERE id = %s"
                cursor.execute(sql, (id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()