W&B TODO
=============

Only save one checkpoint artifact
 - Only save checkpoint at the end
 - Delete checkpoints at the end except for last one

Figure out better groupings and names for W&B runs

Update eval script to handle artifacts and update the artifact that the eval was ran on

Create artifacts endpoints for establishing datasets in W&B

Update train endpoint to either take an artifact name or a path location for the dataset

Track final results to W&B as an artifact

Add identifiers to init of W&B logger callback

break up utils into respective functionalities
 - likely make it its own module

create an artifacts endpoint that allows me to delete W&B runs with a certain tag
