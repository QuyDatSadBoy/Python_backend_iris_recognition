from db_connection import get_connection
from entity.TrainDetectionHistory import TrainDetectionHistory
from typing import List, Optional, Dict

class TrainDetectionHistoryDAO:
    
    @staticmethod
    def getAllTrainDetectionHistory() -> List[TrainDetectionHistory]:
        """Lấy tất cả training history"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                SELECT * FROM tblTrainDetectionHistory
                ORDER BY timeTrain DESC
                """
                cursor.execute(sql)
                results = cursor.fetchall()
                return [TrainDetectionHistory(**result) for result in results]
        finally:
            connection.close()
    
    @staticmethod  
    def getTrainDetectionHistoryById(id: int) -> Optional[TrainDetectionHistory]:
        """Lấy training history theo id"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblTrainDetectionHistory WHERE id = %s"
                cursor.execute(sql, (id,))
                result = cursor.fetchone()
                return TrainDetectionHistory(**result) if result else None
        finally:
            connection.close()
            
    @staticmethod
    def createTrainDetectionHistory(trainDetectionHistory: TrainDetectionHistory) -> int:
        """Tạo training history mới"""
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblTrainDetectionHistory 
                (epochs, batchSize, imageSize, learningRate, 
                 tblDetectEyeDataTrainId, tblEyeDetectionModelId) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    trainDetectionHistory.epochs,
                    trainDetectionHistory.batchSize,
                    trainDetectionHistory.imageSize,
                    trainDetectionHistory.learningRate,
                    trainDetectionHistory.tblDetectEyeDataTrainId,
                    trainDetectionHistory.tblEyeDetectionModelId
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()