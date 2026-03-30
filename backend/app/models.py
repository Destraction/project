from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, JSON, CheckConstraint
from sqlalchemy.sql import func
from app.database import Base

class Ingredient(Base):
    __tablename__ = "ingredients"
    __table_args__ = (
        CheckConstraint("quantity >= 0", name="ck_ingredients_quantity_non_negative"),
        CheckConstraint("quantity <= 5000", name="ck_ingredients_quantity_max_5000"),
        CheckConstraint("price_per_unit >= 0", name="ck_ingredients_price_non_negative"),
    )
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(String(50))  # base, citrus, berry, tropical, spice, herb, sweetener
    unit = Column(String(10), default="ml")
    quantity = Column(Float, default=0.0)  # текущий остаток
    price_per_unit = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Compatibility(Base):
    __tablename__ = "compatibility"
    ing1_id = Column(Integer, ForeignKey("ingredients.id"), primary_key=True)
    ing2_id = Column(Integer, ForeignKey("ingredients.id"), primary_key=True)
    score = Column(Integer, default=1)  # 0=плохо, 1=нейтрально, 2=хорошо

class GeneratedCocktail(Base):
    __tablename__ = "generated_cocktails"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100))
    name = Column(String(200))
    description = Column(Text)
    recipe = Column(JSON)  # {ingredient_id: quantity}
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100))
    cocktail_id = Column(Integer, ForeignKey("generated_cocktails.id"))
    status = Column(String(20), default="pending")  # pending, confirmed, completed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)

class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    rating = Column(Integer)
    comment = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DialogueMessage(Base):
    __tablename__ = "dialogue_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100))
    role = Column(String(20))  # user, assistant
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())