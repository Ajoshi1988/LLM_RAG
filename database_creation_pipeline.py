from dagster import asset, AssetExecutionContext, AssetIn, Definitions, define_asset_job, AssetSelection, ScheduleDefinition


###############Pipeline database creation ###############################

@asset(key="parse_files_key", group_name="VECTOR_DB")
def PARSE_FILES(context: AssetExecutionContext):

    print("I got here")

    return [1,2,3]




@asset(key="create_db_key",ins={"upstream": AssetIn(key="parse_files_key")}, group_name="VECTOR_DB")
def CREATE_DB(context: AssetExecutionContext, upstream: List):
    print("I got here")

    return [1,2,3]



@asset(key="create_embeddings_key",ins={"upstream": AssetIn(key="create_db_key")}, group_name="VECTOR_DB")
def POSTGRE_SINK(context: AssetExecutionContext, upstream: List):
    print("I got here")

    return [1,2,3]




defs=Definitions(
    assets=[PARSE_FILES, CREATE_DB, POSTGRE_SINK ],
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
