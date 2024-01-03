from datetime import datetime
from uuid import uuid4

class Job:
    def __init__(self, data):
        self.id = str(uuid4())
        self.data = data
        self.status = 'queued'  # Possible statuses: queued, running, completed, failed
        self.result = None
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'data': self.data,
            'status': self.status,
            'result': self.result,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
