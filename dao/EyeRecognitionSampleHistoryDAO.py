from db_connection import get_connection
from entity.EyeRecognitionSampleHistory import TrainDetectionHistory
from typing import List, Optional, Dict, Any


class EyeRecognitionSampleHistoryDAO:
    """
    Data Access Object for TrainDetectionHistory
    """
    
    @staticmethod
    def get_all_train_history() -> List[Dict[str, Any]]:
        """
        Get all training history from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                SELECT h.*, m.modelName, m.modelLink, d.dataTrainPath 
                FROM tblTrainDetectionHistory h
                LEFT JOIN tblEyeDetectionModel m ON h.tblEyeDetectionModelId = m.id
                LEFT JOIN tblDetectEyeDataTrain d ON h.tblDetectEyeDataTrainId = d.id
                ORDER BY h.timeTrain DESC
                """
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_train_history_by_id(history_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a training history entry by its ID
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                SELECT h.*, m.modelName, m.modelLink, d.dataTrainPath 
                FROM tblTrainDetectionHistory h
                LEFT JOIN tblEyeDetectionModel m ON h.tblEyeDetectionModelId = m.id
                LEFT JOIN tblDetectEyeDataTrain d ON h.tblDetectEyeDataTrainId = d.id
                WHERE h.id = %s
                """
                cursor.execute(sql, (history_id,))
                result = cursor.fetchone()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_train_history_by_model_id(model_id: int) -> List[Dict[str, Any]]:
        """
        Get all training history entries for a specific model
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                SELECT h.*, m.modelName, m.modelLink, d.dataTrainPath 
                FROM tblTrainDetectionHistory h
                LEFT JOIN tblEyeDetectionModel m ON h.tblEyeDetectionModelId = m.id
                LEFT JOIN tblDetectEyeDataTrain d ON h.tblDetectEyeDataTrainId = d.id
                WHERE h.tblEyeDetectionModelId = %s
                ORDER BY h.timeTrain DESC
                """
                cursor.execute(sql, (model_id,))
                result = cursor.fetchall()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_train_history_by_data_train_id(data_train_id: int) -> List[Dict[str, Any]]:
        """
        Get all training history entries for a specific data training set
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                SELECT h.*, m.modelName, m.modelLink, d.dataTrainPath 
                FROM tblTrainDetectionHistory h
                LEFT JOIN tblEyeDetectionModel m ON h.tblEyeDetectionModelId = m.id
                LEFT JOIN tblDetectEyeDataTrain d ON h.tblDetectEyeDataTrainId = d.id
                WHERE h.tblDetectEyeDataTrainId = %s
                ORDER BY h.timeTrain DESC
                """
                cursor.execute(sql, (data_train_id,))
                result = cursor.fetchall()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def create_train_history(history: TrainDetectionHistory) -> int:
        """
        Create a new training history entry in the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblTrainDetectionHistory 
                (epochs, batchSize, imageSize, learningRate, tblDetectEyeDataTrainId, tblEyeDetectionModelId) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    history.epochs,
                    history.batchSize,
                    history.imageSize,
                    history.learningRate,
                    history.tblDetectEyeDataTrainId,
                    history.tblEyeDetectionModelId
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
    
    @staticmethod
    def update_train_history(history_id: int, history: TrainDetectionHistory) -> bool:
        """
        Update an existing training history entry
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblTrainDetectionHistory 
                SET epochs = %s, batchSize = %s, imageSize = %s, learningRate = %s, 
                    tblDetectEyeDataTrainId = %s, tblEyeDetectionModelId = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    history.epochs,
                    history.batchSize,
                    history.imageSize,
                    history.learningRate,
                    history.tblDetectEyeDataTrainId,
                    history.tblEyeDetectionModelId,
                    history_id
                ))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
    
    @staticmethod
    def delete_train_history(history_id: int) -> bool:
        """
        Delete a training history entry from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "DELETE FROM tblTrainDetectionHistory WHERE id = %s"
                cursor.execute(sql, (history_id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()