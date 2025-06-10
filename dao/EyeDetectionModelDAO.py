from db_connection import get_connection
from entity.EyeDetectionModel import EyeDetectionModel
from typing import List, Optional, Dict

class EyeDetectionModelDAO:
    
    @staticmethod
    def getAllEyeDetectionModel() -> List[EyeDetectionModel]:
        """Lấy tất cả model"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblEyeDetectionModel ORDER BY createDate DESC"
                cursor.execute(sql)
                results = cursor.fetchall()
                return [EyeDetectionModel(**result) for result in results]
        finally:
            connection.close()
    
    @staticmethod
    def getEyeDetectionModelById(id: int) -> Optional[EyeDetectionModel]:
        """Lấy model theo id"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblEyeDetectionModel WHERE id = %s"
                cursor.execute(sql, (id,))
                result = cursor.fetchone()
                return EyeDetectionModel(**result) if result else None
        finally:
            connection.close()
            
    @staticmethod
    def createEyeDetectionModel(eyeDetectionModel: EyeDetectionModel) -> int:
        """Tạo model mới"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblEyeDetectionModel 
                (modelName, mapMetric, isActive, modelLink) 
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    eyeDetectionModel.modelName,
                    eyeDetectionModel.mapMetric,
                    eyeDetectionModel.isActive,
                    eyeDetectionModel.modelLink
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
            
    @staticmethod
    def updateEyeDetectionModel(id: int, eyeDetectionModel: EyeDetectionModel) -> bool:
        """Cập nhật model"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblEyeDetectionModel 
                SET modelName = %s, mapMetric = %s, isActive = %s, modelLink = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    eyeDetectionModel.modelName,
                    eyeDetectionModel.mapMetric,
                    eyeDetectionModel.isActive,
                    eyeDetectionModel.modelLink,
                    id
                ))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
            
    @staticmethod
    def deleteEyeDetectionModel(id: int) -> bool:
        """Xóa model"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                # Xóa các bản ghi tham chiếu trong bảng TrainDetectionHistory trước
                sql_history = "DELETE FROM tblTrainDetectionHistory WHERE tblEyeDetectionModelId = %s"
                cursor.execute(sql_history, (id,))
                
                # Sau đó mới xóa model
                sql = "DELETE FROM tblEyeDetectionModel WHERE id = %s"
                cursor.execute(sql, (id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
            
    @staticmethod
    def setActiveEyeDetectionModel(id: int) -> bool:
        """Đặt model active"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                # Deactivate all
                sql = "UPDATE tblEyeDetectionModel SET isActive = 0"
                cursor.execute(sql)
                
                # Activate selected model
                sql = "UPDATE tblEyeDetectionModel SET isActive = 1 WHERE id = %s"
                cursor.execute(sql, (id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()