
# Import all the models here, so that Base has them before being
# imported by Alembic
from app.db.base_class import Base # noqa
from app.db.models import User, UserPortfolio # noqa
