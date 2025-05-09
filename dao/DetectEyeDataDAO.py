from db_connection import get_connection
from entity.DetectEyeData import DetectEyeData
from typing import List, Optional, Dict

class DetectEyeDataDAO:
    
    @staticmethod
    def getAllDetectEyeData(detectEyeDataTrainId: int) -> List[DetectEyeData]:
        """Lấy tất cả eye data theo training data id"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeData WHERE tblDetectEyeDataTrainId = %s"
                cursor.execute(sql, (detectEyeDataTrainId,))
                results = cursor.fetchall()
                return [DetectEyeData(**result) for result in results]
        finally:
            connection.close()
    
    @staticmethod
    def getDetectEyeDataById(id: int) -> Optional[DetectEyeData]:
        """Lấy eye data theo id"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblDetectEyeData WHERE id = %s"
                cursor.execute(sql, (id,))
                result = cursor.fetchone()
                return DetectEyeData(**result) if result else None
        finally:
            connection.close()
            
    @staticmethod
    def createDetectEyeData(detectEyeData: DetectEyeData) -> int:
        """Tạo eye data mới"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblDetectEyeData 
                (imageLink, labelLink, tblDetectEyeDataTrainId) 
                VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (
                    detectEyeData.imageLink,
                    detectEyeData.labelLink,
                    detectEyeData.tblDetectEyeDataTrainId
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
            
    @staticmethod
    def createDetectEyeDataList(detectEyeDataList: List[DetectEyeData]) -> bool:
        """Tạo nhiều eye data"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblDetectEyeData 
                (imageLink, labelLink, tblDetectEyeDataTrainId) 
                VALUES (%s, %s, %s)
                """
                data = [(item.imageLink, item.labelLink, item.tblDetectEyeDataTrainId) 
                       for item in detectEyeDataList]
                cursor.executemany(sql, data)
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
            
    @staticmethod
    def deleteDetectEyeData(id: int) -> bool:
        """Xóa eye data"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "DELETE FROM tblDetectEyeData WHERE id = %s"
                cursor.execute(sql, (id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()