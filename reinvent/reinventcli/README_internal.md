REINVENT 2.0
=======================================================================================

For contributors
-----
Below are some guidelines that would be good to follow when contributing
to the project. 

* General 
    1. This code aims to be object oriented. Try to adhere as much as possible to the
     SOLID principles.
    2. Avoid piling up multiple classes in a single file.
    3. Avoid using static methods. If you do, make sure you call them from the instance of
     the class.
    4. Use explicit naming for classes, methods, variables and etc. This saves the need of
     extensive documentation.
    5. If you find yourself having to write explanations what your code does, ask yourself 
    "why?". There is a good chance you just need to simplify your code and name everything 
    properly.
    6. Please, make an effort to provide decent code and case coverage in your unit tests.
    7. Use typing. Help the IDE so that the IDE can help you.
    8. Dont use magic strings. We can turn a blind eye on strings contained within the 
    class but those should never leave the scope of the class.
    9. For unit tests best follow the Arrange, Act, Assert pattern.

* Specific
    1. Wherever possible use .get() on dictionaries.
    2. Don't reuse variable names within the same scope. This just obfuscates things.
    3. Don't return different types of data that doesnt even share the same interface.
    Just don`t.
    4. Avoid returning tuples of objects wherever not needed, and especially from 
    public methods.
    5. Don't test private methods.
    6. Use __ init__.py files for indexing and don't pile up logic there.
    7. Dont use proprietary SMILES strings in the unit test. Best use some well publicly
    known compounds.


__When in doubt, ask!__

Usage
-----

1. Sample inputs are provided in reinvent/configs/sample_inputs folder.

2. (Recommended) Use the GUI to generate inputs.

-------------------------------------------------
To use Tensorboard for logging:

   1. To launch Tensorboard, you need a graphical environment like VDI. Write:
       tensorboard --logdir "path to your log output directory" --port=8008.
       This will give you an address to copy to a browser and access to the graphical summaries from Tensorboard.

   2. Further commands to Tensorboard to change the amount of scalars,histograms, images, distributions and graphs shown
        can be done as follows:
        --samples_per_plugin=scalar=700, images=20

Installation
-----

1. Install Anaconda / Miniconda
2. Clone the repository
3. (Optional) Checkout the appropriate branch of the repository and create a new local branch tied to the remote one, e.g.:
    git checkout --track origin/reinvent.v.2
4. Open terminal, go to the repository and generate the appropriate environment:
    conda env create -f reinvent.yml
   Hint: Use the appropriate `conda` binary. You might want to check, whether you succeeded:
    conda info --envs
5. You will need to set the environmental variable OE_LICENSE to activate the oechem license. One way to do this and keep it conda environment specific is:
   On the command line, first:

       cd $CONDA_PREFIX
       mkdir -p ./etc/conda/activate.d
       mkdir -p ./etc/conda/deactivate.d
       touch ./etc/conda/activate.d/env_vars.sh
       touch ./etc/conda/deactivate.d/env_vars.sh

   then edit ./etc/conda/activate.d/env_vars.sh as follows:

       #!/bin/sh


   and finally, edit ./etc/conda/deactivate.d/env_vars.sh :

       #!/bin/sh
       unset OE_LICENSE
6. Activate environment (or set it in your GUI)
7. In the project directory, in ./configs/ create the file `config.json` by copying over `example.config.json` and editing as required


Tests
-----
The tests can be executed using Unittest:
```
python -m unittest
```

Or using Pytest:
```
python -m pytest unittest_reinvent
```

Integration tests are decorated with `@pytest.mark.integration`. You can easily skip integration tests using pytest mark expression (`-m` argument):
```
python -m pytest -m "not integration" --strict-markers unittest_reinvent/
```
