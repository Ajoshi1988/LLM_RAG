from dagster import asset, AssetExecutionContext, AssetIn, Definitions, define_asset_job, AssetSelection, ScheduleDefinition
from typing import List

###############Pipeline database creation ###############################

@asset(key="PARSE_FILES", group_name="VECTOR_DB")
def PARSE_FILES(context: AssetExecutionContext):

    print("I got here")

    return [1,2,3]




@asset(key="CREATE_SCHEMA",ins={"upstream": AssetIn(key="PARSE_FILES")}, group_name="VECTOR_DB")
def CREATE_DB(context: AssetExecutionContext, upstream: List):
    print("I got here")

    return [1,2,3]



@asset(key="POSTGRESQL_SINK",ins={"upstream": AssetIn(key="CREATE_SCHEMA")}, group_name="VECTOR_DB")
def POSTGRE_SINK(context: AssetExecutionContext, upstream: List):
    print("I got here")

    return [1,2,3]


@asset(key="DSPY_FINE_TUNING",ins={"upstream": AssetIn(key="POSTGRESQL_SINK")}, group_name="VECTOR_DB")
def DSPY(context: AssetExecutionContext, upstream: List):
    print("I got here")

    return [1,2,3]



defs=Definitions(
    assets=[PARSE_FILES, CREATE_DB, POSTGRE_SINK, DSPY ],
    jobs=[
        define_asset_job(

            name="Data_embeddings_job",
            selection=AssetSelection.groups("VECTOR_DB"),
        )

         ],

    schedules=[

        ScheduleDefinition(

            name="Data_Embeddings_Schedule",
            job_name="Data_embeddings_job",
            cron_schedule="*/10 * * * *"
        )
    ]

)
