#####
Tasks
#####

Pippin is essentially a wrapper around many different tasks. In this section, I'll try and explain how tasks are related to each other, and what each task is.

As a general note, most tasks have an ``OPTS`` section where most details go. This is partially historical, but essentially properties that Pippin uses to determine how to construct tasks (like ``MASK``, classification mode, etc) are top level, and the Task itself gets passed everything inside OPTS to use however it wants.

.. toctree::
    :maxdepth: 1

    tasks/dataprep
