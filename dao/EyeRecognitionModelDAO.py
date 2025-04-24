from db_connection import get_connection
from entity.EyeRecognitionModel import EyeRecognitionModel
from typing import List, Optional, Dict, Any
import os


class EyeRecognitionModelDAO:
    """
    Data Access Object for EyeRecognitionModel
    """
    
    @staticmethod
    def get_all_models() -> List[Dict[str, Any]]:
        """
        Get all models from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblEyeDetectionModel ORDER BY createDate DESC"
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_model_by_id(model_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a model by its ID
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblEyeDetectionModel WHERE id = %s"
                cursor.execute(sql, (model_id,))
                result = cursor.fetchone()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def get_active_model() -> Optional[Dict[str, Any]]:
        """
        Get the currently active model
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM tblEyeDetectionModel WHERE isActive = 1 ORDER BY createDate DESC LIMIT 1"
                cursor.execute(sql)
                result = cursor.fetchone()
                return result
        finally:
            connection.close()
    
    @staticmethod
    def create_model(model: EyeRecognitionModel) -> int:
        """
        Create a new model in the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO tblEyeDetectionModel 
                (modelName, mapMetric, isActive, modelLink) 
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    model.modelName,
                    model.mapMetric,
                    model.isActive,
                    model.modelLink
                ))
                connection.commit()
                return cursor.lastrowid
        finally:
            connection.close()
    
    @staticmethod
    def update_model(model_id: int, model: EyeRecognitionModel) -> bool:
        """
        Update an existing model
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblEyeDetectionModel 
                SET modelName = %s, mapMetric = %s, isActive = %s, modelLink = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    model.modelName,
                    model.mapMetric,
                    model.isActive,
                    model.modelLink,
                    model_id
                ))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
    
    @staticmethod
    def delete_model(model_id: int) -> bool:
        """
        Delete a model from the database
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                # First, check if the model file exists and delete it
                sql = "SELECT modelLink FROM tblEyeDetectionModel WHERE id = %s"
                cursor.execute(sql, (model_id,))
                result = cursor.fetchone()
                
                if result:
                    # Convert relative path to absolute path
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    model_path = os.path.join(base_dir, result['modelLink'])
                    
                    if os.path.exists(model_path):
                        os.remove(model_path)
                
                # Then delete from the database
                sql = "DELETE FROM tblEyeDetectionModel WHERE id = %s"
                cursor.execute(sql, (model_id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()
    
    @staticmethod
    def set_active_model(model_id: int) -> bool:
        """
        Set a model as active and deactivate all others
        """
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                # Deactivate all models
                sql = "UPDATE tblEyeDetectionModel SET isActive = 0"
                cursor.execute(sql)
                
                # Activate the selected model
                sql = "UPDATE tblEyeDetectionModel SET isActive = 1 WHERE id = %s"
                cursor.execute(sql, (model_id,))
                connection.commit()
                return cursor.rowcount > 0
        finally:
            connection.close()