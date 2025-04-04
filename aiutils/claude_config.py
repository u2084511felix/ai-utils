import anthropic
from dotenv import load_dotenv
import os
from configs.helpers import extract_python_code


load_dotenv()

my_api_key = os.getenv("ANTHROPIC_API_KEY")


client = anthropic.Anthropic(
    api_key=my_api_key
)

prompt1 = "please list the processes associated with the generic dataclasses. (on first build of the app backened system, a generate process should generate the data and update the databases for example)"
prompt2 = "The generic data will be displayed in different places in the app such as milestones stracker, etc.. some generic data is vital to the apps core functionality actually. Please list all of these processes."

process_example = """import json
from configs.openai_config import create_app_data
from db.schema import DevelopmentalStage

async def generate_developmental_stages(session):
    # Initial instruction
    instruction = "Generate a list of developmental stages for babies based on standard age ranges and milestones."

    # Context
    instruction += "The developmental stages should cover the following age ranges: 0-1 month, 1-3 months, 3-6 months, 6-9 months, 9-12 months, 12-18 months, 18-24 months, and 24+ months."
    instruction += "For each developmental stage, provide a name, description, and the corresponding age range."

    # Response Format Instructions
    instruction += "Please provide the developmental stages in the following JSON format:"
    instruction += '''
    [
        {
            "name": "Developmental Stage Name",
            "description": "A brief description of the developmental stage, including key milestones and characteristics.",
            "age_range": "The age range covered by the developmental stage, e.g., 0-1 month, 1-3 months, etc."
        }
    ]
    '''

    # AI GENERATION
    developmental_stages_json = await create_app_data("You are a child development expert.", instruction)
    developmental_stages = json.loads(developmental_stages_json)
    print(developmental_stages)

    for stage in developmental_stages:
        developmental_stage = DevelopmentalStage(
            name=stage["name"],
            description=stage["description"],
            age_range=stage["age_range"]
        )
        session.add(developmental_stage)

    session.flush()
    session.commit()

async def update_developmental_stages(session):
    # Initial instruction
    instruction = "Update the list of developmental stages for babies based on the latest research and guidelines."

    # Context
    instruction += "The developmental stages should cover the following age ranges: 0-1 month, 1-3 months, 3-6 months, 6-9 months, 9-12 months, 12-18 months, 18-24 months, and 24+ months."
    instruction += "For each developmental stage, provide an updated name, description, and the corresponding age range."
    instruction += "If there are no updates for a particular developmental stage, please include the stage with its existing information."

    # Response Format Instructions
    instruction += "Please provide the updated developmental stages in the following JSON format:"
    instruction += '''
    [
        {
            "name": "Updated Developmental Stage Name",
            "description": "An updated brief description of the developmental stage, including key milestones and characteristics.",
            "age_range": "The age range covered by the developmental stage, e.g., 0-1 month, 1-3 months, etc."
        }
    ]
    '''

    # AI GENERATION
    updated_stages_json = await create_app_data("You are a child development expert.", instruction)
    updated_stages = json.loads(updated_stages_json)
    print(updated_stages)

    for stage in updated_stages:
        developmental_stage = session.query(DevelopmentalStage).filter_by(age_range=stage["age_range"]).first()
        if developmental_stage:
            developmental_stage.name = stage["name"]
            developmental_stage.description = stage["description"]
        else:
            developmental_stage = DevelopmentalStage(
                name=stage["name"],
                description=stage["description"],
                age_range=stage["age_range"]
            )
            session.add(developmental_stage)

    session.flush()
    session.commit()"""


app_roadmap = """Babu App Features & Roadmap


Initial Profile Creation and Setup

    • Basic Information Entry
        ◦ The mother enters the babys basic information: sex, age (or due/birth date), and weight. This initial data sets the context for the development stage and initial content personalization.
    • Development Stage Assessment
        ◦ Based on the age provided, the app automatically identifies the baby's current developmental stage, tailoring the subsequent questions and content to suit this stage.
    • Milestone Achievement Inquiry
        ◦ The app prompts the mother with a checklist of milestones typical for the baby's current developmental stage to identify which milestones have been reached, aiding in further personalization of content and recommendations.
    • Medical Conditions/Needs Discussion
        ◦ A series of questions about any known medical conditions, allergies, or special needs the baby has, ensuring that any care recommendations take these into account.
    • Premium subscription: Babies tastes / temperament profile.
        ◦ The ability to add and track a likes dislikes list to the babies profile.
        ◦ The ability to add and track a temperament section to the app.
        ◦ Tailored content, routines, recipes, and activities to match the temperament and tastes of the baby.
    • First time users will be prompted to set up essential routines
        ◦ These will include both the sleep and feeding routines initially.

Building the Sleep and Feeding Routine

    • Sleep Routine Establishment
        ◦ The app asks detailed questions about the baby's current sleep patterns, preferences, and any challenges faced. This includes questions about sleep durations, night awakenings, and nap times. It then suggests a suitable routine, which the mother can accept or adapt.
    • Feeding Routine Formation
        ◦ Inquiries about the babys feeding schedule, including breastfeeding, formula feeding, and, if applicable, solid food introduction. It also gathers information on any dietary restrictions, feeding challenges, or preferences. It then suggests a suitable feeding routine, which the mother can accept or adapt. It will also suggest recipes and a balanced diet plan for the baby and mother.

Customization and Support

    • Care categories:
        ◦ Essentials categories: Sleeping and Feeding.
        ◦ Additional categories: Health / Medical & Hygiene, Safety Tips, Parental Care, Developmental Activities / Play and Bonding.
        ◦ Each category will have suggestions of activities, routines, and have curated resources / content.
    • Suggestions Review and Adjustment Mechanism 
        ◦ Each accepted suggestion made by the app will be listed under the relevant categories.
        ◦ Accepted suggestions will have a feedback mechanism – with the mother providing the babies response or her own response. 
        ◦ Based on the category, adjustments will be made to the babies temperament and likes / dislikes profile, and new suggestions will be made to better suit the baby and caregivers.
        ◦ A notification system for accepted routines will be active by default but can be turned off.
    • Resource / Content Provision 
        ◦ The app will suggest essential and recommended resources for the mother and baby.
        ◦ It will also curate or generate articles, and tips tailored to the baby's / mothers developmental stage, tastes / behaviour profile, achieved milestones, and any specific needs or conditions mentioned, ensuring parents have access to supportive and informative content.

Feature Interaction Widgets and Mechanisms
    • Routine and Milestone Tracking
        ◦ Tracking sleep and feeding patterns, milestone achievements, and growth metrics, encouraging regular updates for ongoing personalization and support.
    • Periodic Reviews and Updates
        ◦ Periodically prompts the mother to review and update the babys profile with any new milestones, changes in routines, or medical conditions, ensuring the apps recommendations and content stay relevant as the baby grows.
    • Chatbot interface
        ◦ A chatbot with up to date profile knowledge about the baby and mother, will provide a general inquiry service tailored to answering questions related to the babies care needs.

------

    • Daily Summary Widget: Offers a snapshot of the day's scheduled feedings, sleep times, and any special notes or reminders. It could adapt based on the baby's routines and any input from the previous days.
      
    • Milestone Tracker Widget: Displays upcoming developmental milestones based on the baby's age and any previously achieved milestones. This could include prompts for parents to watch for new skills or behaviours.
      
    • Feeding Timer Widget: A quick-start timer for breastfeeding or bottle-feeding sessions, tracking duration and side (for breastfeeding), with the option to log the feeding once it's completed.
      
    • Sleep Quality Overview Widget: Summarizes the baby's sleep from the previous night and naps from the current day, providing insights or tips to improve sleep based on the tracked data.
      
    • Health and Vaccination Reminder Widget: Lists upcoming doctor's appointments, vaccination schedules, and health check reminders based on the baby's age and medical history.
      
    • Quick Tips Widget: Offers daily parenting tips, activities, or fun facts tailored to the baby's developmental stage and interests, possibly integrating content from the premium subscription features like temperament and tastes profiles.
      
    • Growth Tracker Widget: Visual graphs or summaries showing the baby's growth metrics (height, weight, head circumference) against standard paediatric charts, with reminders for parents to log new measurements.
      
    • Community Access Widget: Direct link to a community forum or support group within the app, where parents can share experiences, ask questions, and find support from a network of peers.
      
    • Personalized Content Feed Widget: A dynamic feed of articles, videos, and resources tailored to the baby's developmental stage, milestones achieved, and any special needs or interests noted in their profile.
      
    • Quick Access to Chatbot Widget: Ensuring the chatbot icon is prominently placed on the home screen for easy access to personalized advice and answers to frequently asked questions.



Pre-Natal Stage Content:

    • Pregnancy Milestones: Have you reached specific pregnancy milestones? (e.g., first heartbeat, first movement, trimester stages) This helps in providing timely and relevant information.
      
    • Emotional Well-being: How do you currently feel emotionally? (e.g., excited, anxious, stressed) Understanding emotional states can guide the delivery of supportive mental health resources.
      
    • Learning Interests: What pregnancy-related topics are you most interested in learning about? (e.g., nutrition, exercise, labor and delivery, baby care) Directs the app to prioritize content based on specific interests.
      
    • Sleep Habits: How would you describe your current sleep quality and patterns? Sleep is crucial during pregnancy, and insights can lead to personalized sleep hygiene tips.
      
    • Support System: Do you feel you have a supportive network? (e.g., partner, family, friends, online community) Suggestions for enhancing support networks or providing resources can be made based on the answer.
      
    • Baby's Nursery Preparation: Have you started preparing your baby's nursery? Guidance on nursery essentials and safety can be offered.
      
    • Baby Name Ideas: Are you looking for baby name ideas? This could lead to fun, engaging content about baby names.
      
    • Maternity Leave Planning: Have you started planning your maternity leave? Information on maternity rights and planning tips can be provided.
      
    • Birth Plan Preferences: Have you thought about your birth plan? (e.g., natural, C-section, epidural) Offering information on different birth options and what to expect can be valuable.
      
    • Classes and Workshops: Are you interested in prenatal classes or workshops? (e.g., childbirth, breastfeeding, parenting) Recommendations for online or local resources can be made.
      
    • Baby Gear and Product Preferences: Do you have any preferences or needs for baby products? (e.g., eco-friendly, budget-friendly, specific brands) Customized recommendations and reviews can be provided.
      
    • Cultural or Religious Considerations: Are there cultural or religious considerations in your pregnancy or baby care plans? Tailoring content to respect and incorporate these aspects.
      
    • Previous Pregnancy Experiences: If not a first pregnancy, how does this pregnancy compare to your previous ones? Insights into unique concerns or comparisons can be addressed.
      
    • Concerns and Questions: What are your biggest concerns or questions about pregnancy and childbirth? Directly addressing common concerns through curated content or expert advice.
"""

app_ui_blueprint = """Project Overview:

The Baby Bumps App is a comprehensive, personalized app designed to support mothers and caregivers throughout their baby's development journey. The app provides tailored routines, milestone tracking, and expert content based on the baby's unique profile, including age, developmental stage, medical needs, and preferences.

User Personas:
1. New mothers seeking guidance and support in their baby's first year.
2. Experienced mothers looking for personalized tips and resources.
3. Caregivers (fathers, grandparents, nannies) involved in the baby's daily care.

Key Features and UI Design Instructions:

1. Onboarding and Profile Setup:
   - Design a simple, intuitive form for mothers to input their baby's sex, age (or due/birth date), and weight.
   - Create a visual representation of the baby's automatically identified developmental stage based on the provided age.
   - Design an interactive checklist for mothers to mark off achieved milestones, typical for the baby's current developmental stage.
   - Develop a series of user-friendly questions to gather information about the baby's medical conditions, allergies, or special needs.
   - For premium subscribers, include additional profile sections for tracking the baby's tastes, likes/dislikes, and temperament.

2. Routine Building:
   - Create detailed questionnaires to gather information about the baby's sleep patterns, preferences, challenges, feeding schedules, and any dietary restrictions or preferences.
   - Design intuitive interfaces for suggesting and customizing sleep and feeding routines based on the gathered information.
   - Develop visually appealing displays for recommended recipes and balanced diet plans.

3. Care Categories and Content:
   - Design distinct sections for each care category: Sleeping, Feeding, Health/Medical & Hygiene, Safety Tips, Parental Care, and Developmental Activities/Play and Bonding.
   - Create a feedback mechanism for mothers to provide input on their baby's response to suggested routines and activities, and design an intuitive way to display adjustments made based on this feedback.
   - Develop a visually appealing way to present curated articles, tips, and resources tailored to the baby's profile and needs.

4. Feature Interaction Widgets:
   - Design clear, concise, and visually engaging widgets for the home screen, including Daily Summary, Milestone Tracker, Feeding Timer, Sleep Quality Overview, Health and Vaccination Reminders, Quick Tips, Growth Tracker, Community Access, Personalized Content Feed, and Quick Access to Chatbot.
   - Ensure each widget provides relevant, personalized information based on the baby's profile and user interactions.

5. Pre-Natal Stage Content:
   - Design engaging questionnaires or input forms to gather information about the mother's pregnancy milestones, emotional well-being, learning interests, sleep habits, support system, nursery preparation, baby name ideas, maternity leave planning, birth plan preferences, interest in classes and workshops, baby gear and product preferences, cultural or religious considerations, previous pregnancy experiences, and concerns or questions.
   - Create visually appealing and informative content displays tailored to the mother's specific needs and interests based on her input.
"""


app_schema = """Here's a schema design for the Baby Bumps App based on the provided app description and design brief:```pythonimport typing
from sqlalchemy import Column, DateTime, Integer, String, Float, Date, Time, ForeignKey, Enum, CheckConstraint, Table, MetaData, Float, UniqueConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime
from sqlalchemy.orm import relationship, sessionmaker, Mapped, mapped_column, declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True, index=True, unique=True)
    username: Mapped[str] = mapped_column(nullable=False)
    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[Enum] = mapped_column(Enum('mother', 'caregiver', name='user_role_enum'))
    baby: Mapped["Baby"] = relationship("Baby", back_populates="user", uselist=False)
    subscription: Mapped["Subscription"] = relationship("Subscription", back_populates="user", uselist=False)
    feedback: Mapped[typing.List["Feedback"]] = relationship("Feedback", back_populates="user")
    interactions: Mapped[typing.List["Interaction"]] = relationship("Interaction", back_populates="user")
    pregnancy_milestones: Mapped[typing.List["PregnancyMilestone"]] = relationship("PregnancyMilestone", back_populates="user")
    emotional_wellbeing: Mapped[typing.List["EmotionalWellbeing"]] = relationship("EmotionalWellbeing", back_populates="user")
    learning_interests: Mapped[typing.List["LearningInterest"]] = relationship("LearningInterest", back_populates="user")
    sleep_habits: Mapped[typing.List["SleepHabit"]] = relationship("SleepHabit", back_populates="user")
    support_systems: Mapped[typing.List["SupportSystem"]] = relationship("SupportSystem", back_populates="user")
    nursery_preparations: Mapped[typing.List["NurseryPreparation"]] = relationship("NurseryPreparation", back_populates="user")
    baby_name_ideas: Mapped[typing.List["BabyNameIdea"]] = relationship("BabyNameIdea", back_populates="user")
    maternity_leaves: Mapped[typing.List["MaternityLeave"]] = relationship("MaternityLeave", back_populates="user")
    birth_plans: Mapped[typing.List["BirthPlan"]] = relationship("BirthPlan", back_populates="user")
    class_workshops: Mapped[typing.List["ClassWorkshop"]] = relationship("ClassWorkshop", back_populates="user")
    baby_gear_preferences: Mapped[typing.List["BabyGearPreference"]] = relationship("BabyGearPreference", back_populates="user")
    cultural_religious_considerations: Mapped[typing.List["CulturalReligiousConsideration"]] = relationship("CulturalReligiousConsideration", back_populates="user")
    previous_pregnancy_experiences: Mapped[typing.List["PreviousPregnancyExperience"]] = relationship("PreviousPregnancyExperience", back_populates="user")
    concern_questions: Mapped[typing.List["ConcernQuestion"]] = relationship("ConcernQuestion", back_populates="user")

class Baby(Base):
    __tablename__ = 'babies'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    name: Mapped[str] = mapped_column(nullable=True)
    sex: Mapped[Enum] = mapped_column(Enum('male', 'female', name='baby_sex_enum'))
    birth_date: Mapped[Date] = mapped_column(nullable=False)
    weight: Mapped[float] = mapped_column(nullable=True)
    developmental_stage_id: Mapped[int] = mapped_column(ForeignKey('developmental_stages.id'))
    user: Mapped["User"] = relationship("User", back_populates="baby")
    developmental_stage: Mapped["DevelopmentalStage"] = relationship("DevelopmentalStage", back_populates="babies")
    milestones: Mapped[typing.List["Milestone"]] = relationship("Milestone", secondary="baby_milestones", back_populates="babies")
    medical_conditions: Mapped[typing.List["MedicalCondition"]] = relationship("MedicalCondition", back_populates="baby")
    allergies: Mapped[typing.List["Allergy"]] = relationship("Allergy", back_populates="baby")
    special_needs: Mapped[typing.List["SpecialNeed"]] = relationship("SpecialNeed", back_populates="baby")
    tastes: Mapped[typing.List["Taste"]] = relationship("Taste", back_populates="baby")
    likes_dislikes: Mapped[typing.List["LikeDislike"]] = relationship("LikeDislike", back_populates="baby")
    temperament: Mapped[typing.List["Temperament"]] = relationship("Temperament", back_populates="baby")
    sleep_patterns: Mapped[typing.List["SleepPattern"]] = relationship("SleepPattern", back_populates="baby")
    feeding_schedules: Mapped[typing.List["FeedingSchedule"]] = relationship("FeedingSchedule", back_populates="baby")
    dietary_restrictions: Mapped[typing.List["DietaryRestriction"]] = relationship("DietaryRestriction", back_populates="baby")
    sleep_routine: Mapped["SleepRoutine"] = relationship("SleepRoutine", back_populates="baby", uselist=False)
    feeding_routine: Mapped["FeedingRoutine"] = relationship("FeedingRoutine", back_populates="baby", uselist=False)
    vaccinations: Mapped[typing.List["Vaccination"]] = relationship("Vaccination", secondary="baby_vaccinations", back_populates="babies")
    health_checkups: Mapped[typing.List["HealthCheckup"]] = relationship("HealthCheckup", secondary="baby_health_checkups", back_populates="babies")

class Milestone(Base):
    __tablename__ = 'milestones'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    milestone: Mapped[str] = mapped_column(nullable=False)
    achieved: Mapped[bool] = mapped_column(default=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="milestones")


class MedicalCondition(Base):
    __tablename__ = 'medical_conditions'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    common_medical_condition_id: Mapped[int] = mapped_column(ForeignKey('common_medical_conditions.id'))
    baby: Mapped["Baby"] = relationship("Baby", back_populates="medical_conditions")
    common_medical_condition: Mapped["CommonMedicalCondition"] = relationship("CommonMedicalCondition", back_populates="medical_conditions")

class Allergy(Base):
    __tablename__ = 'allergies'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    common_allergy_id: Mapped[int] = mapped_column(ForeignKey('common_allergies.id'))
    baby: Mapped["Baby"] = relationship("Baby", back_populates="allergies")
    common_allergy: Mapped["CommonAllergy"] = relationship("CommonAllergy", back_populates="allergies")

class SpecialNeed(Base):
    __tablename__ = 'special_needs'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    need: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="special_needs")

class Taste(Base):
    __tablename__ = 'tastes'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    taste: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="tastes")

class LikeDislike(Base):
    __tablename__ = 'likes_dislikes'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    item: Mapped[str] = mapped_column(nullable=False)
    preference: Mapped[Enum] = mapped_column(Enum('like', 'dislike', name='preference_enum'))
    baby: Mapped["Baby"] = relationship("Baby", back_populates="likes_dislikes")

class Temperament(Base):
    __tablename__ = 'temperaments'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    temperament: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="temperament")

class SleepPattern(Base):
    __tablename__ = 'sleep_patterns'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    pattern: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="sleep_patterns")


class FeedingSchedule(Base):
    __tablename__ = 'feeding_schedules'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    schedule: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="feeding_schedules")
    nutrients: Mapped[typing.List["Nutrient"]] = relationship("Nutrient", secondary="feeding_schedule_nutrients", back_populates="feeding_schedules")
    foods: Mapped[typing.List["Food"]] = relationship("Food", secondary="feeding_schedule_foods", back_populates="feeding_schedules")

class DietaryRestriction(Base):
    __tablename__ = 'dietary_restrictions'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    restriction: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="dietary_restrictions")


class SleepRoutine(Base):
    __tablename__ = 'sleep_routines'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    routine: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="sleep_routine")

class FeedingRoutine(Base):
    __tablename__ = 'feeding_routines'
    id: Mapped[int] = mapped_column(primary_key=True)
    baby_id: Mapped[int] = mapped_column(ForeignKey('babies.id'))
    routine: Mapped[str] = mapped_column(nullable=False)
    baby: Mapped["Baby"] = relationship("Baby", back_populates="feeding_routine")

class Subscription(Base):
    __tablename__ = 'subscriptions'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    type: Mapped[Enum] = mapped_column(Enum('free', 'premium', name='subscription_type_enum'))
    user: Mapped["User"] = relationship("User", back_populates="subscription")

class Feedback(Base):
    __tablename__ = 'feedback'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    content: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="feedback")

class Interaction(Base):
    __tablename__ = 'interactions'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    type: Mapped[str] = mapped_column(nullable=False)
    data: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="interactions")

class Content(Base):
    __tablename__ = 'content'
    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[Enum] = mapped_column(Enum('sleeping', 'feeding', 'health_medical_hygiene', 'safety_tips', 'parental_care', 'developmental_activities_play_bonding', name='content_category_enum'))
    title: Mapped[str] = mapped_column(nullable=False)
    body: Mapped[str] = mapped_column(nullable=False)
    tags: Mapped[typing.List["ContentTag"]] = relationship("ContentTag", back_populates="content")


class ContentTag(Base):
    __tablename__ = 'content_tags'
    id: Mapped[int] = mapped_column(primary_key=True)
    content_id: Mapped[int] = mapped_column(ForeignKey('content.id'))
    tag: Mapped[str] = mapped_column(nullable=False)
    content: Mapped["Content"] = relationship("Content", back_populates="tags")


class PregnancyMilestone(Base):
    __tablename__ = 'pregnancy_milestones'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    pregnancy_stage_id: Mapped[int] = mapped_column(ForeignKey('pregnancy_stages.id'))
    milestone: Mapped[str] = mapped_column(nullable=False)
    achieved: Mapped[bool] = mapped_column(default=False)
    user: Mapped["User"] = relationship("User", back_populates="pregnancy_milestones")
    pregnancy_stage: Mapped["PregnancyStage"] = relationship("PregnancyStage", back_populates="pregnancy_milestones")

class EmotionalWellbeing(Base):
    __tablename__ = 'emotional_wellbeing'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    status: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="emotional_wellbeing")

class LearningInterest(Base):
    __tablename__ = 'learning_interests'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    interest: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="learning_interests")

class SleepHabit(Base):
    __tablename__ = 'sleep_habits'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    habit: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="sleep_habits")

class SupportSystem(Base):
    __tablename__ = 'support_systems'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    support: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="support_systems")

class NurseryPreparation(Base):
    __tablename__ = 'nursery_preparations'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    preparation: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="nursery_preparations")

class BabyNameIdea(Base):
    __tablename__ = 'baby_name_ideas'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    name: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="baby_name_ideas")

class MaternityLeave(Base):
    __tablename__ = 'maternity_leaves'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    plan: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="maternity_leaves")


class BirthPlan(Base):
    __tablename__ = 'birth_plans'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    birth_option_id: Mapped[int] = mapped_column(ForeignKey('birth_options.id'))
    preference: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="birth_plans")
    birth_option: Mapped["BirthOption"] = relationship("BirthOption", back_populates="birth_plans")

class ClassWorkshop(Base):
    __tablename__ = 'class_workshops'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    interest: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="class_workshops")


class BabyGearPreference(Base):
    __tablename__ = 'baby_gear_preferences'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    preference: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="baby_gear_preferences")
    categories: Mapped[typing.List["BabyGearCategory"]] = relationship("BabyGearCategory", secondary="baby_gear_preference_categories", back_populates="baby_gear_preferences")
    brands: Mapped[typing.List["BabyGearBrand"]] = relationship("BabyGearBrand", secondary="baby_gear_preference_brands", back_populates="baby_gear_preferences")

class CulturalReligiousConsideration(Base):
    __tablename__ = 'cultural_religious_considerations'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    consideration: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="cultural_religious_considerations")

class PreviousPregnancyExperience(Base):
    __tablename__ = 'previous_pregnancy_experiences'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    experience: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="previous_pregnancy_experiences")

class ConcernQuestion(Base):
    __tablename__ = 'concern_questions'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    concern_question: Mapped[str] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="concern_questions")

class DevelopmentalStage(Base):
    __tablename__ = 'developmental_stages'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    babies: Mapped[typing.List["Baby"]] = relationship("Baby", back_populates="developmental_stage")
    milestones: Mapped[typing.List["Milestone"]] = relationship("Milestone", back_populates="developmental_stage")

class EssentialItem(Base):
    __tablename__ = 'essential_items'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_essential_items", back_populates="essential_items")

class Vaccination(Base):
    __tablename__ = 'vaccinations'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    babies: Mapped[typing.List["Baby"]] = relationship("Baby", secondary="baby_vaccinations", back_populates="vaccinations")

class HealthCheckup(Base):
    __tablename__ = 'health_checkups'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    babies: Mapped[typing.List["Baby"]] = relationship("Baby", secondary="baby_health_checkups", back_populates="health_checkups")

class CommonAllergy(Base):
    __tablename__ = 'common_allergies'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    symptoms: Mapped[str] = mapped_column(nullable=False)
    allergies: Mapped[typing.List["Allergy"]] = relationship("Allergy", back_populates="common_allergy")

class CommonMedicalCondition(Base):
    __tablename__ = 'common_medical_conditions'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    symptoms: Mapped[str] = mapped_column(nullable=False)
    medical_conditions: Mapped[typing.List["MedicalCondition"]] = relationship("MedicalCondition", back_populates="common_medical_condition")

class Nutrient(Base):
    __tablename__ = 'nutrients'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    recommended_daily_intake: Mapped[str] = mapped_column(nullable=False)
    feeding_schedules: Mapped[typing.List["FeedingSchedule"]] = relationship("FeedingSchedule", secondary="feeding_schedule_nutrients", back_populates="nutrients")

class Food(Base):
    __tablename__ = 'foods'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    serving_size: Mapped[str] = mapped_column(nullable=False)
    feeding_schedules: Mapped[typing.List["FeedingSchedule"]] = relationship("FeedingSchedule", secondary="feeding_schedule_foods", back_populates="foods")

class Recipe(Base):
    __tablename__ = 'recipes'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    ingredients: Mapped[str] = mapped_column(nullable=False)
    instructions: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_recipes", back_populates="recipes")

class Activity(Base):
    __tablename__ = 'activities'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_activities", back_populates="activities")

class SafetyTip(Base):
    __tablename__ = 'safety_tips'
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_safety_tips", back_populates="safety_tips")

class ParentalCareResource(Base):
    __tablename__ = 'parental_care_resources'
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    tags: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_parental_care_resources", back_populates="parental_care_resources")

class PregnancyStage(Base):
    __tablename__ = 'pregnancy_stages'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    duration: Mapped[str] = mapped_column(nullable=False)
    pregnancy_milestones: Mapped[typing.List["PregnancyMilestone"]] = relationship("PregnancyMilestone", back_populates="pregnancy_stage")

class PregnancySymptom(Base):
    __tablename__ = 'pregnancy_symptoms'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    remedies: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_pregnancy_symptoms", back_populates="pregnancy_symptoms")

class BirthOption(Base):
    __tablename__ = 'birth_options'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    pros: Mapped[str] = mapped_column(nullable=False)
    cons: Mapped[str] = mapped_column(nullable=False)
    birth_plans: Mapped[typing.List["BirthPlan"]] = relationship("BirthPlan", back_populates="birth_option")

class PostpartumCareResource(Base):
    __tablename__ = 'postpartum_care_resources'
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    tags: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_postpartum_care_resources", back_populates="postpartum_care_resources")

class BabyGearCategory(Base):
    __tablename__ = 'baby_gear_categories'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    baby_gear_preferences: Mapped[typing.List["BabyGearPreference"]] = relationship("BabyGearPreference", secondary="baby_gear_preference_categories", back_populates="categories")

class BabyGearBrand(Base):
    __tablename__ = 'baby_gear_brands'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    website: Mapped[str] = mapped_column(nullable=False)
    baby_gear_preferences: Mapped[typing.List["BabyGearPreference"]] = relationship("BabyGearPreference", secondary="baby_gear_preference_brands", back_populates="brands")

class ParentingTip(Base):
    __tablename__ = 'parenting_tips'
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(nullable=False)
    age_range: Mapped[str] = mapped_column(nullable=False)
    users: Mapped[typing.List["User"]] = relationship("User", secondary="user_parenting_tips", back_populates="parenting_tips")

# Association tables
baby_milestones = Table(
    'baby_milestones',
    Base.metadata,
    Column('baby_id', ForeignKey('babies.id'), primary_key=True),
    Column('milestone_id', ForeignKey('milestones.id'), primary_key=True)
)

baby_vaccinations = Table(
    'baby_vaccinations',
    Base.metadata,
    Column('baby_id', ForeignKey('babies.id'), primary_key=True),
    Column('vaccination_id', ForeignKey('vaccinations.id'), primary_key=True)
)

baby_health_checkups = Table(
    'baby_health_checkups',
    Base.metadata,
    Column('baby_id', ForeignKey('babies.id'), primary_key=True),
    Column('health_checkup_id', ForeignKey('health_checkups.id'), primary_key=True)
)

feeding_schedule_nutrients = Table(
    'feeding_schedule_nutrients',
    Base.metadata,
    Column('feeding_schedule_id', ForeignKey('feeding_schedules.id'), primary_key=True),
    Column('nutrient_id', ForeignKey('nutrients.id'), primary_key=True)
)

feeding_schedule_foods = Table(
    'feeding_schedule_foods',
    Base.metadata,
    Column('feeding_schedule_id', ForeignKey('feeding_schedules.id'), primary_key=True),
    Column('food_id', ForeignKey('foods.id'), primary_key=True)
)

user_essential_items = Table(
    'user_essential_items',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('essential_item_id', ForeignKey('essential_items.id'), primary_key=True)
)

user_recipes = Table(
    'user_recipes',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('recipe_id', ForeignKey('recipes.id'), primary_key=True)
)

user_activities = Table(
    'user_activities',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('activity_id', ForeignKey('activities.id'), primary_key=True)
)

user_safety_tips = Table(
    'user_safety_tips',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('safety_tip_id', ForeignKey('safety_tips.id'), primary_key=True)
)

user_parental_care_resources = Table(
    'user_parental_care_resources',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('parental_care_resource_id', ForeignKey('parental_care_resources.id'), primary_key=True)
)

user_pregnancy_symptoms = Table(
    'user_pregnancy_symptoms',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('pregnancy_symptom_id', ForeignKey('pregnancy_symptoms.id'), primary_key=True)
)

user_postpartum_care_resources = Table(
    'user_postpartum_care_resources',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('postpartum_care_resource_id', ForeignKey('postpartum_care_resources.id'), primary_key=True)
)

baby_gear_preference_categories = Table(
    'baby_gear_preference_categories',
    Base.metadata,
    Column('baby_gear_preference_id', ForeignKey('baby_gear_preferences.id'), primary_key=True),
    Column('baby_gear_category_id', ForeignKey('baby_gear_categories.id'), primary_key=True)
)

baby_gear_preference_brands = Table(
    'baby_gear_preference_brands',
    Base.metadata,
    Column('baby_gear_preference_id', ForeignKey('baby_gear_preferences.id'), primary_key=True),
    Column('baby_gear_brand_id', ForeignKey('baby_gear_brands.id'), primary_key=True)
)

user_parenting_tips = Table(
    'user_parenting_tips',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('parenting_tip_id', ForeignKey('parenting_tips.id'), primary_key=True)
)"
                }
            ]
        },
    ]
)
print(message.content)"""

schema_relations = """The relationships between the generic data classes and the user-specific classes in the database schema:

1. DevelopmentalStage:
   - A baby belongs to a specific developmental stage based on their age.
   - Relationship: Baby (many) -> DevelopmentalStage (one)

2. Milestone:
   - A baby can achieve multiple milestones, and each milestone is associated with a specific developmental stage.
   - Relationship: Baby (many) -> Milestone (many)
   - Relationship: Milestone (many) -> DevelopmentalStage (one)

3. EssentialItem:
   - A user can mark essential items as purchased or needed for their baby.
   - Relationship: User (many) -> EssentialItem (many) (through a join table, e.g., UserEssentialItem)

4. Vaccination:
   - A baby can receive multiple vaccinations, and each vaccination is associated with a specific age range.
   - Relationship: Baby (many) -> Vaccination (many) (through a join table, e.g., BabyVaccination)

5. HealthCheckup:
   - A baby can have multiple health checkups, and each health checkup is associated with a specific age range.
   - Relationship: Baby (many) -> HealthCheckup (many) (through a join table, e.g., BabyHealthCheckup)

6. CommonAllergy:
   - A baby can have multiple allergies, and each allergy is associated with a specific common allergy.
   - Relationship: Baby (many) -> CommonAllergy (many) (through the Allergy class)

7. CommonMedicalCondition:
   - A baby can have multiple medical conditions, and each condition is associated with a specific common medical condition.
   - Relationship: Baby (many) -> CommonMedicalCondition (many) (through the MedicalCondition class)

8. Nutrient:
   - A baby's feeding schedule can include multiple nutrients, and each nutrient is associated with a specific recommended daily intake.
   - Relationship: FeedingSchedule (many) -> Nutrient (many) (through a join table, e.g., FeedingScheduleNutrient)

9. Food:
   - A baby's feeding schedule can include multiple foods, and each food is associated with a specific age range.
   - Relationship: FeedingSchedule (many) -> Food (many) (through a join table, e.g., FeedingScheduleFood)

10. Recipe:
    - A user can save multiple recipes for their baby, and each recipe is associated with a specific age range.
    - Relationship: User (many) -> Recipe (many) (through a join table, e.g., UserRecipe)

11. Activity:
    - A user can save multiple activities for their baby, and each activity is associated with a specific age range.
    - Relationship: User (many) -> Activity (many) (through a join table, e.g., UserActivity)

12. SafetyTip:
    - A user can save multiple safety tips, and each safety tip is associated with a specific age range.
    - Relationship: User (many) -> SafetyTip (many) (through a join table, e.g., UserSafetyTip)

13. ParentalCareResource:
    - A user can save multiple parental care resources, and each resource is associated with specific tags.
    - Relationship: User (many) -> ParentalCareResource (many) (through a join table, e.g., UserParentalCareResource)

14. PregnancyStage:
    - A user's pregnancy milestone can be associated with a specific pregnancy stage.
    - Relationship: PregnancyMilestone (many) -> PregnancyStage (one)

15. PregnancySymptom:
    - A user can experience multiple pregnancy symptoms, and each symptom is associated with a specific common pregnancy symptom.
    - Relationship: User (many) -> PregnancySymptom (many) (through a join table, e.g., UserPregnancySymptom)

16. BirthOption:
    - A user's birth plan can include a specific birth option.
    - Relationship: BirthPlan (many) -> BirthOption (one)

17. PostpartumCareResource:
    - A user can save multiple postpartum care resources, and each resource is associated with specific tags.
    - Relationship: User (many) -> PostpartumCareResource (many) (through a join table, e.g., UserPostpartumCareResource)

18. BabyGearCategory:
    - A user's baby gear preferences can include multiple gear categories.
    - Relationship: BabyGearPreference (many) -> BabyGearCategory (many) (through a join table, e.g., BabyGearPreferenceCategory)

19. BabyGearBrand:
    - A user's baby gear preferences can include multiple gear brands.
    - Relationship: BabyGearPreference (many) -> BabyGearBrand (many) (through a join table, e.g., BabyGearPreferenceBrand)

20. ParentingTip:
    - A user can save multiple parenting tips, and each tip is associated with a specific age range.
    - Relationship: User (many) -> ParentingTip (many) (through a join table, e.g., UserParentingTip)

These relationships establish the connections between the generic data classes and the user-specific classes in the database schema. They  associate relevant information from the generic classes with specific users, babies, and their preferences, milestones, and activities.

By utilizing these relationships, we can efficiently store and retrieve data, provide personalized recommendations, and generate tailored content based on the unique characteristics and needs of each user and their baby.
"""

app_processes = """Processes related to displaying and utilizing generic data in different parts of the app:

1. Milestone Tracking:
   - Retrieve relevant milestones based on the baby's current developmental stage.
   - Display the milestones in the milestone tracker section of the app.
   - Allow users to mark milestones as achieved and update the baby's profile accordingly.
   - Provide recommendations for activities or resources based on the achieved milestones.

2. Vaccination Tracking:
   - Retrieve the list of recommended vaccinations based on the baby's age and vaccination history.
   - Display the vaccination schedule in the app's vaccination tracker section.
   - Allow users to mark vaccinations as completed and update the baby's profile accordingly.
   - Provide reminders for upcoming vaccinations based on the baby's age and the standard vaccination schedule.

3. Health Checkup Tracking:
   - Retrieve the list of recommended health checkups based on the baby's age and checkup history.
   - Display the health checkup schedule in the app's health checkup tracker section.
   - Allow users to mark health checkups as completed and update the baby's profile accordingly.
   - Provide reminders for upcoming health checkups based on the baby's age and the standard checkup schedule.

4. Feeding and Nutrition:
   - Retrieve age-appropriate foods, nutrients, and recipes based on the baby's age and dietary preferences.
   - Display the recommended foods, nutrients, and recipes in the app's feeding and nutrition section.
   - Allow users to track the baby's food intake and update the baby's profile accordingly.
   - Provide personalized feeding recommendations based on the baby's age, dietary preferences, and nutritional requirements.

5. Activity Suggestions:
   - Retrieve age-appropriate activities based on the baby's developmental stage and preferences.
   - Display the suggested activities in the app's activity section.
   - Allow users to mark activities as completed and provide feedback on the baby's engagement and enjoyment.
   - Provide personalized activity recommendations based on the baby's developmental stage, preferences, and feedback.

6. Safety Tips:
   - Retrieve relevant safety tips based on the baby's age and developmental stage.
   - Display the safety tips in the app's safety section.
   - Allow users to mark safety tips as read and provide feedback on their usefulness.
   - Provide personalized safety recommendations based on the baby's age, developmental stage, and user feedback.

7. Parental Care Resources:
   - Retrieve relevant parental care resources based on the user's preferences and the baby's age.
   - Display the parental care resources in the app's resource section.
   - Allow users to save or bookmark helpful resources for future reference.
   - Provide personalized resource recommendations based on the user's preferences, the baby's age, and user feedback.

8. Pregnancy Tracking:
   - Retrieve relevant pregnancy stages, symptoms, and milestones based on the user's due date or pregnancy progress.
   - Display the pregnancy information in the app's pregnancy tracking section.
   - Allow users to track their pregnancy symptoms, weight gain, and other relevant data.
   - Provide personalized pregnancy recommendations based on the user's progress, symptoms, and preferences.

9. Baby Gear Recommendations:
   - Retrieve relevant baby gear categories and brands based on the user's preferences and the baby's age.
   - Display the baby gear recommendations in the app's gear section.
   - Allow users to save or bookmark preferred baby gear items for future reference.
   - Provide personalized baby gear recommendations based on the user's preferences, the baby's age, and user feedback.

10. Parenting Tips:
    - Retrieve relevant parenting tips based on the baby's age and the user's preferences.
    - Display the parenting tips in the app's parenting section.
    - Allow users to save or bookmark helpful parenting tips for future reference.
    - Provide personalized parenting tip recommendations based on the baby's age, user preferences, and feedback.

These processes involve retrieving relevant generic data from the database, combining it with user-specific data, and presenting it in a meaningful way within the app's various sections. The app should provide a seamless user experience by displaying the most relevant and personalized information based on the baby's profile and the user's preferences.

Additionally, there should be processes to handle user interactions with the displayed data, such as allowing users to mark items as completed, provide feedback, save preferences, and receive personalized recommendations based on their actions and feedback.



Core app processes:

1. Create user profile: Process to create a new user profile with basic information, subscription details, and pre-natal stage content.

2. Update user profile: Process to update user profile information, including personal details, preferences, and pre-natal stage content.

3. Create baby profile: Process to create a new baby profile with basic information, developmental stage, milestones, medical conditions, allergies, special needs, tastes, likes/dislikes, and temperament.

4. Update baby profile: Process to update baby profile information, including developmental stage, milestones, medical conditions, allergies, special needs, tastes, likes/dislikes, and temperament.

5. Generate sleep routine: Process to generate a personalized sleep routine for the baby based on the baby's profile, sleep patterns, and user preferences.

6. Generate feeding routine: Process to generate a personalized feeding routine for the baby based on the baby's profile, feeding schedules, dietary restrictions, and user preferences.

7. Update sleep routine: Process to update the baby's sleep routine based on user feedback and baby's response.

8. Update feeding routine: Process to update the baby's feeding routine based on user feedback and baby's response.

9. Track milestones: Process to track and update the baby's achieved milestones based on user input.

10. Provide milestone suggestions: Process to suggest upcoming milestones based on the baby's age and development stage.

11. Generate personalized content: Process to generate and suggest personalized content (articles, tips, resources) based on the baby's profile, developmental stage, and user preferences.

12. Update content preferences: Process to update user's content preferences based on their interactions and feedback.

13. Provide health and vaccination reminders: Process to generate and send health and vaccination reminders based on the baby's age and medical history.

14. Update medical history: Process to update the baby's medical history based on user input and completed vaccinations.

15. Provide activity suggestions: Process to suggest age-appropriate activities and play ideas based on the baby's developmental stage and preferences.

16. Update activity preferences: Process to update the baby's activity preferences based on user feedback and baby's response.

17. Generate growth tracker: Process to generate and update the baby's growth tracker based on user input of height, weight, and head circumference measurements.

18. Provide community recommendations: Process to recommend relevant community forums or support groups based on the user's profile and preferences.

19. Handle user feedback: Process to receive, store, and analyze user feedback on various aspects of the app.

20. Generate chatbot responses: Process to generate personalized chatbot responses based on the user's profile, baby's profile, and the context of the user's question or concern.

21. Update taste preferences: Process to update the baby's taste preferences based on user feedback and baby's response to new foods.

22. Suggest recipes: Process to suggest age-appropriate recipes based on the baby's taste preferences, dietary restrictions, and feeding schedule.

23. Provide safety tips: Process to provide personalized safety tips based on the baby's developmental stage and user's preferences.

24. Update parental care preferences: Process to update the user's parental care preferences based on their feedback and experiences.

25. Suggest parental care resources: Process to suggest parental care resources (articles, videos, support groups) based on the user's preferences and emotional well-being.

26. Generate pre-natal content: Process to generate and suggest pre-natal content based on the user's pregnancy stage, milestones, and preferences.

27. Update pre-natal preferences: Process to update the user's pre-natal content preferences based on their feedback and interests.

This list covers the main processes required for the baby app based on the provided schema and blueprint documents. Each process involves data retrieval, data manipulation, and data storage operations, as well as potential interactions with external services or APIs for generating personalized content and recommendations.



Processes associated with the generic data classes:

1. Developmental Stage:
   - Generate default developmental stages based on standard age ranges and milestones.
   - Update developmental stage information as new research or guidelines become available.

2. Essential Item:
   - Generate a list of essential items for babies based on general recommendations.
   - Update essential item information as new products or guidelines become available.

3. Vaccination:
   - Generate a list of recommended vaccinations based on standard vaccination schedules.
   - Update vaccination information as new vaccines or guidelines become available.

4. Health Checkup:
   - Generate a list of recommended health checkups based on standard pediatric guidelines.
   - Update health checkup information as new guidelines or recommendations become available.

5. Common Allergy:
   - Generate a list of common allergies in babies based on medical research and data.
   - Update common allergy information as new research or data becomes available.

6. Common Medical Condition:
   - Generate a list of common medical conditions in babies based on medical research and data.
   - Update common medical condition information as new research or data becomes available.

7. Nutrient:
   - Generate a list of essential nutrients for babies based on standard nutritional guidelines.
   - Update nutrient information as new research or guidelines become available.

8. Food:
   - Generate a list of age-appropriate foods for babies based on standard feeding guidelines.
   - Update food information as new research or guidelines become available.

9. Recipe:
   - Generate a list of age-appropriate recipes for babies based on standard feeding guidelines and nutritional requirements.
   - Update recipe information as new recipes or guidelines become available.

10. Activity:
    - Generate a list of age-appropriate activities for babies based on developmental milestones and guidelines.
    - Update activity information as new research or guidelines become available.

11. Safety Tip:
    - Generate a list of safety tips for babies based on standard safety guidelines and recommendations.
    - Update safety tip information as new guidelines or recommendations become available.

12. Parental Care Resource:
    - Generate a list of parental care resources based on common parenting topics and concerns.
    - Update parental care resource information as new resources or topics become available.

13. Pregnancy Stage:
    - Generate a list of pregnancy stages based on standard gestational timelines and milestones.
    - Update pregnancy stage information as new research or guidelines become available.

14. Pregnancy Symptom:
    - Generate a list of common pregnancy symptoms based on medical research and data.
    - Update pregnancy symptom information as new research or data becomes available.

15. Birth Option:
    - Generate a list of common birth options based on standard childbirth practices and guidelines.
    - Update birth option information as new options or guidelines become available.

16. Postpartum Care Resource:
    - Generate a list of postpartum care resources based on common postpartum topics and concerns.
    - Update postpartum care resource information as new resources or topics become available.

17. Baby Gear Category:
    - Generate a list of baby gear categories based on common baby product classifications.
    - Update baby gear category information as new categories or classifications become available.

18. Baby Gear Brand:
    - Generate a list of popular baby gear brands based on market research and consumer data.
    - Update baby gear brand information as new brands emerge or gain popularity.

19. Parenting Tip:
    - Generate a list of parenting tips based on common parenting challenges and best practices.
    - Update parenting tip information as new tips or best practices become available.

These processes involve generating initial data for the generic data classes based on standard guidelines, research, and recommendations. The data should be periodically updated to ensure it remains current and accurate.

Additionally, there may be processes to handle user interactions with the generic data, such as:

- Retrieving relevant generic data based on user preferences or baby's profile.
- Filtering and sorting generic data based on user-specified criteria.
- Providing personalized recommendations by combining user-specific data with relevant generic data.

These processes will involve querying the database, applying business logic, and presenting the data to the user in a meaningful way."""


more_app_features_string = """Here's a comprehensive and organized list of processes and features for the baby app, combining and abstracting similar parts to reduce redundancy:

1. User Management:
   - Create user profile
   - Update user profile
   - Manage user preferences
   - Handle user feedback

2. Baby Profile Management:
   - Create baby profile
   - Update baby profile
   - Track baby's developmental stage
   - Update baby's medical history
   - Update baby's preferences (tastes, likes/dislikes, temperament)

3. Personalized Routines:
   - Generate sleep routine
   - Update sleep routine
   - Generate feeding routine
   - Update feeding routine

4. Milestone Tracking:
   - Track achieved milestones
   - Suggest upcoming milestones
   - Provide milestone-related content and resources

5. Health and Vaccination Management:
   - Provide health and vaccination reminders
   - Track completed vaccinations
   - Provide vaccination-related content and resources

6. Activity Suggestions:
   - Suggest age-appropriate activities
   - Update activity preferences
   - Provide activity-related content and resources

7. Nutrition and Feeding:
   - Suggest age-appropriate foods and nutrients
   - Update taste preferences
   - Suggest recipes based on preferences and restrictions
   - Provide nutrition-related content and resources

8. Growth Tracking:
   - Generate growth tracker
   - Update growth measurements (height, weight, head circumference)
   - Provide growth-related content and resources

9. Safety and Parental Care:
   - Provide personalized safety tips
   - Suggest parental care resources
   - Update parental care preferences

10. Pre-natal Care:
    - Generate pre-natal content based on pregnancy stage
    - Update pre-natal preferences
    - Provide pregnancy-related content and resources

11. Community and Support:
    - Recommend relevant community forums or support groups
    - Facilitate user interaction and engagement

12. Personalized Content Generation:
    - Generate personalized articles, tips, and resources based on user and baby profiles
    - Update content preferences based on user interactions and feedback

13. Chatbot Interaction:
    - Generate personalized chatbot responses based on user and baby profiles
    - Handle user inquiries and provide relevant information

14. Baby Gear Recommendations:
    - Suggest baby gear based on user preferences and baby's age
    - Provide baby gear-related content and resources

15. Generic Data Management:
    - Generate and update generic data for developmental stages, essential items, vaccinations, health checkups, common allergies, medical conditions, nutrients, foods, recipes, activities, safety tips, parental care resources, pregnancy stages, symptoms, birth options, postpartum care resources, baby gear categories, and parenting tips
    - Retrieve and filter generic data based on user preferences and baby's profile
    - Provide personalized recommendations by combining user-specific and generic data

16. Data Analytics and Insights:
    - Analyze user interactions and feedback to improve app features and content
    - Generate insights and reports on user behavior and preferences
    - Identify trends and patterns in baby development and parenting practices

17. Notifications and Reminders:
    - Send personalized notifications and reminders for milestones, vaccinations, health checkups, and other important events
    - Allow users to customize notification preferences

18. Integration with External Services:
    - Integrate with healthcare providers or medical records systems for seamless data exchange
    - Integrate with e-commerce platforms for baby gear purchases
    - Integrate with social media platforms for user engagement and sharing

19. Security and Privacy:
    - Implement secure user authentication and authorization
    - Protect user data through encryption and secure storage
    - Comply with relevant privacy regulations and standards

20. Performance and Scalability:
    - Optimize app performance for fast loading times and smooth user experience
    - Implement scalable infrastructure to handle growing user base and data volume
    - Conduct regular performance testing and monitoring

This organized list provides a comprehensive overview of the processes and features required for the baby app. It combines similar functionalities into logical groups, abstracts common processes, and ensures a complete and cohesive user experience. Developers can use this as a guide to plan and implement the various components of the app while maintaining a focus on personalization, data management, and user engagement."""


more_app_features = [
    "User Management: Create user profile, Update user profile, Manage user preferences, Handle user feedback",
    "Baby Profile Management: Create baby profile, Update baby profile, Track baby's developmental stage, Update baby's medical history, Update baby's preferences (tastes, likes/dislikes, temperament)",
    "Personalized Routines: Generate sleep routine, Update sleep routine, Generate feeding routine, Update feeding routine",
    "Milestone Tracking: Track achieved milestones, Suggest upcoming milestones, Provide milestone-related content and resources",
    "Health and Vaccination Management: Provide health and vaccination reminders, Track completed vaccinations, Provide vaccination-related content and resources",
    "Activity Suggestions: Suggest age-appropriate activities, Update activity preferences, Provide activity-related content and resources",
    "Nutrition and Feeding: Suggest age-appropriate foods and nutrients, Update taste preferences, Suggest recipes based on preferences and restrictions, Provide nutrition-related content and resources",
    "Growth Tracking: Generate growth tracker, Update growth measurements (height, weight, head circumference), Provide growth-related content and resources",
    "Safety and Parental Care: Provide personalized safety tips, Suggest parental care resources, Update parental care preferences",
    "Pre-natal Care: Generate pre-natal content based on pregnancy stage, Update pre-natal preferences, Provide pregnancy-related content and resources",
    "Community and Support: Recommend relevant community forums or support groups, Facilitate user interaction and engagement",
    "Personalized Content Generation: Generate personalized articles, tips, and resources based on user and baby profiles, Update content preferences based on user interactions and feedback",
    "Chatbot Interaction: Generate personalized chatbot responses based on user and baby profiles, Handle user inquiries and provide relevant information",
    "Baby Gear Recommendations: Suggest baby gear based on user preferences and baby's age, Provide baby gear-related content and resources",
    "Generic Data Management: Generate and update generic data for developmental stages, essential items, vaccinations, health checkups, common allergies, medical conditions, nutrients, foods, recipes, activities, safety tips, parental care resources, pregnancy stages, symptoms, birth options, postpartum care resources, baby gear categories, and parenting tips, Retrieve and filter generic data based on user preferences and baby's profile, Provide personalized recommendations by combining user-specific and generic data",
    "Data Analytics and Insights: Analyze user interactions and feedback to improve app features and content, Generate insights and reports on user behavior and preferences, Identify trends and patterns in baby development and parenting practices",
    "Notifications and Reminders: Send personalized notifications and reminders for milestones, vaccinations, health checkups, and other important events, Allow users to customize notification preferences",
    "Integration with External Services: Integrate with healthcare providers or medical records systems for seamless data exchange, Integrate with e-commerce platforms for baby gear purchases, Integrate with social media platforms for user engagement and sharing",
    "Security and Privacy: Implement secure user authentication and authorization, Protect user data through encryption and secure storage, Comply with relevant privacy regulations and standards",
    "Performance and Scalability: Optimize app performance for fast loading times and smooth user experience, Implement scalable infrastructure to handle growing user base and data volume, Conduct regular performance testing and monitoring"
]




processes = [
    "Create user profile: Process to create a new user profile with basic information, subscription details, and pre-natal stage content.",
    "Update user profile: Process to update user profile information, including personal details, preferences, and pre-natal stage content.",
    "Create baby profile: Process to create a new baby profile with basic information, developmental stage, milestones, medical conditions, allergies, special needs, tastes, likes/dislikes, and temperament.",
    "Update baby profile: Process to update baby profile information, including developmental stage, milestones, medical conditions, allergies, special needs, tastes, likes/dislikes, and temperament.",
    "Generate sleep routine: Process to generate a personalized sleep routine for the baby based on the baby's profile, sleep patterns, and user preferences.",
    "Generate feeding routine: Process to generate a personalized feeding routine for the baby based on the baby's profile, feeding schedules, dietary restrictions, and user preferences.",
    "Update sleep routine: Process to update the baby's sleep routine based on user feedback and baby's response.",
    "Update feeding routine: Process to update the baby's feeding routine based on user feedback and baby's response.",
    "Track milestones: Process to track and update the baby's achieved milestones based on user input.",
    "Provide milestone suggestions: Process to suggest upcoming milestones based on the baby's age and development stage.",
    "Generate personalized content: Process to generate and suggest personalized content (articles, tips, resources) based on the baby's profile, developmental stage, and user preferences.",
    "Update content preferences: Process to update user's content preferences based on their interactions and feedback.",
    "Provide health and vaccination reminders: Process to generate and send health and vaccination reminders based on the baby's age and medical history.",
    "Update medical history: Process to update the baby's medical history based on user input and completed vaccinations.",
    "Provide activity suggestions: Process to suggest age-appropriate activities and play ideas based on the baby's developmental stage and preferences.",
    "Update activity preferences: Process to update the baby's activity preferences based on user feedback and baby's response.",
    "Generate growth tracker: Process to generate and update the baby's growth tracker based on user input of height, weight, and head circumference measurements.",
    "Provide community recommendations: Process to recommend relevant community forums or support groups based on the user's profile and preferences.",
    "Handle user feedback: Process to receive, store, and analyze user feedback on various aspects of the app.",
    "Generate chatbot responses: Process to generate personalized chatbot responses based on the user's profile, baby's profile, and the context of the user's question or concern.",
    "Update taste preferences: Process to update the baby's taste preferences based on user feedback and baby's response to new foods.",
    "Suggest recipes: Process to suggest age-appropriate recipes based on the baby's taste preferences, dietary restrictions, and feeding schedule.",
    "Provide safety tips: Process to provide personalized safety tips based on the baby's developmental stage and user's preferences.",
    "Update parental care preferences: Process to update the user's parental care preferences based on their feedback and experiences.",
    "Suggest parental care resources: Process to suggest parental care resources (articles, videos, support groups) based on the user's preferences and emotional well-being.",
    "Generate pre-natal content: Process to generate and suggest pre-natal content based on the user's pregnancy stage, milestones, and preferences.",
    "Update pre-natal preferences: Process to update the user's pre-natal content preferences based on their feedback and interests."
]

features = [
    "Milestone Tracking: Retrieve relevant milestones based on the baby's current developmental stage. Display the milestones in the milestone tracker section of the app. Allow users to mark milestones as achieved and update the baby's profile accordingly. Provide recommendations for activities or resources based on the achieved milestones.",
    "Vaccination Tracking: Retrieve the list of recommended vaccinations based on the baby's age and vaccination history. Display the vaccination schedule in the app's vaccination tracker section. Allow users to mark vaccinations as completed and update the baby's profile accordingly. Provide reminders for upcoming vaccinations based on the baby's age and the standard vaccination schedule.",
    "Health Checkup Tracking: Retrieve the list of recommended health checkups based on the baby's age and checkup history. Display the health checkup schedule in the app's health checkup tracker section. Allow users to mark health checkups as completed and update the baby's profile accordingly. Provide reminders for upcoming health checkups based on the baby's age and the standard checkup schedule.",
    "Feeding and Nutrition: Retrieve age-appropriate foods, nutrients, and recipes based on the baby's age and dietary preferences. Display the recommended foods, nutrients, and recipes in the app's feeding and nutrition section. Allow users to track the baby's food intake and update the baby's profile accordingly. Provide personalized feeding recommendations based on the baby's age, dietary preferences, and nutritional requirements.",
    "Activity Suggestions: Retrieve age-appropriate activities based on the baby's developmental stage and preferences. Display the suggested activities in the app's activity section. Allow users to mark activities as completed and provide feedback on the baby's engagement and enjoyment. Provide personalized activity recommendations based on the baby's developmental stage, preferences, and feedback.",
    "Safety Tips: Retrieve relevant safety tips based on the baby's age and developmental stage. Display the safety tips in the app's safety section. Allow users to mark safety tips as read and provide feedback on their usefulness. Provide personalized safety recommendations based on the baby's age, developmental stage, and user feedback.",
    "Parental Care Resources: Retrieve relevant parental care resources based on the user's preferences and the baby's age. Display the parental care resources in the app's resource section. Allow users to save or bookmark helpful resources for future reference. Provide personalized resource recommendations based on the user's preferences, the baby's age, and user feedback.",
    "Pregnancy Tracking: Retrieve relevant pregnancy stages, symptoms, and milestones based on the user's due date or pregnancy progress. Display the pregnancy information in the app's pregnancy tracking section. Allow users to track their pregnancy symptoms, weight gain, and other relevant data. Provide personalized pregnancy recommendations based on the user's progress, symptoms, and preferences.",
    "Baby Gear Recommendations: Retrieve relevant baby gear categories and brands based on the user's preferences and the baby's age. Display the baby gear recommendations in the app's gear section. Allow users to save or bookmark preferred baby gear items for future reference. Provide personalized baby gear recommendations based on the user's preferences, the baby's age, and user feedback.",
    "Parenting Tips: Retrieve relevant parenting tips based on the baby's age and the user's preferences. Display the parenting tips in the app's parenting section. Allow users to save or bookmark helpful parenting tips for future reference. Provide personalized parenting tip recommendations based on the baby's age, user preferences, and feedback."
]

generic_data_processes = [
    "Developmental Stage:- Generate default developmental stages based on standard age ranges and milestones.- Update developmental stage information as new research or guidelines become available.",
    "Essential Item:- Generate a list of essential items for babies based on general recommendations.- Update essential item information as new products or guidelines become available.",
    "Vaccination:- Generate a list of recommended vaccinations based on standard vaccination schedules.- Update vaccination information as new vaccines or guidelines become available.",
    "Health Checkup:- Generate a list of recommended health checkups based on standard pediatric guidelines.- Update health checkup information as new guidelines or recommendations become available.",
    "Common Allergy:- Generate a list of common allergies in babies based on medical research and data.- Update common allergy information as new research or data becomes available.",
    "Common Medical Condition:- Generate a list of common medical conditions in babies based on medical research and data.- Update common medical condition information as new research or data becomes available.",
    "Nutrient:- Generate a list of essential nutrients for babies based on standard nutritional guidelines.- Update nutrient information as new research or guidelines become available.",
    "Food:- Generate a list of age-appropriate foods for babies based on standard feeding guidelines.- Update food information as new research or guidelines become available.",
    "Recipe:- Generate a list of age-appropriate recipes for babies based on standard feeding guidelines and nutritional requirements.- Update recipe information as new recipes or guidelines become available.",
    "Activity:- Generate a list of age-appropriate activities for babies based on developmental milestones and guidelines.- Update activity information as new research or guidelines become available.",
    "Safety Tip:- Generate a list of safety tips for babies based on standard safety guidelines and recommendations.- Update safety tip information as new guidelines or recommendations become available.",
    "Parental Care Resource:- Generate a list of parental care resources based on common parenting topics and concerns.- Update parental care resource information as new resources or topics become available.",
    "Pregnancy Stage:- Generate a list of pregnancy stages based on standard gestational timelines and milestones.- Update pregnancy stage information as new research or guidelines become available.",
    "Pregnancy Symptom:- Generate a list of common pregnancy symptoms based on medical research and data.- Update pregnancy symptom information as new research or data becomes available.",
    "Birth Option:- Generate a list of common birth options based on standard childbirth practices and guidelines.- Update birth option information as new options or guidelines become available.",
    "Postpartum Care Resource:- Generate a list of postpartum care resources based on common postpartum topics and concerns.- Update postpartum care resource information as new resources or topics become available.",
    "Baby Gear Category:- Generate a list of baby gear categories based on common baby product classifications.- Update baby gear category information as new categories or classifications become available.",
    "Baby Gear Brand:- Generate a list of popular baby gear brands based on market research and consumer data.- Update baby gear brand information as new brands emerge or gain popularity.",
    "Parenting Tip:- Generate a list of parenting tips based on common parenting challenges and best practices.- Update parenting tip information as new tips or best practices become available."
]


logic_prompt = f"Given the the following app processes and feature lists, please evaluate the list, remove redunancies, combine processes, come up with some abstractions for meta processes for similar parts, and provide a more comprehensive list of processes and features for the app, so that I have a single list that is both well organized, complete and not unecesarrily repetetive. When I say comprehensive I mean this should be the developers guide, so each process should be mapped out in some detail and the arrangement of processes should be arranged to work together... Here are the app process lists I have made previously: {app_processes}"


async def gen_new_proc_logic():
    message = await generate(logic_prompt)

    print("Generating new process logic...")

    with open(f'./procs/new_procs_readme2.md', 'w') as f:
        f.write(message.content[0].text)
    


gen_pprocess_rompt = f"Please give me the script for <process> Remember also that it should be have a precise instruction string explaining what data to generate, and how the json response should be formatted. Developmental stages should be 0-1 month, 1-3, 3-6, 6-9, 9-12, 12-18, 18-24, 24+"


async def gen_proc_loop():
    i = 0
    for process in more_app_features:
        i += 1
        print("Generating process script for proc ", i)
        print("\n...\n")
        process_prompt = f"Please give me the script for this process: {process}. Make sure to consider existing logic and functions in the app and to reference everything correctly and appropriately. Come up with full implementations of the code solutions."
        message = await generate(process_prompt)
        print("Finished process script for proc ", i)
        print()

        with open(f'./temp/claude/core_proc{i}.md', 'w') as f:
            f.write(message.content[0].text)

        code = extract_python_code(message.content[0].text)

        with open(f'./procs/core/core_proc{i}.py', 'w') as f:
            f.write(code)


async def generate_process_script(process_prompt):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=0,
        system="You are a Data Engineering expert and python programming expert. Do not leave any missing data out of the requested improvement.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": app_roadmap + app_ui_blueprint + app_schema + schema_relations + prompt1 + prompt2 + "Generative AI process example: " + process_example + " Please give me some core app features."
                    }
                ]
            },
                    {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": more_app_features_string
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": process_prompt
                    }
                ]
            }
        ]
    )
    return message




async def generate(prompt):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=0,
        system="You are a Data Engineering expert and python programming expert. Do not leave any missing data out of the requested improvement.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message
