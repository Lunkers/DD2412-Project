To run the tests install pytest:

    pip install pytest

If modulenotfound error occurs, just use the following from the root directory of the project to run the tests:

    python -m pytest tests/

All files, classes and methods with prefix test, \_test or test\_ will be executed and output will be something like this if all tests passes:

    platform win32 -- Python 3.7.6, pytest-6.1.1, py-1.9.0, pluggy-0.13.1
    rootdir: <directory of project>
    collected 20 items                                                                                 
    
    tests\test_layers.py ....................                                                    [100%]
    
    ======================================== warnings summary =========================================
    .env\lib\site-packages\win32\lib\pywintypes.py:2
      C:\KTH\DL\project\DD2412-Project\.env\lib\site-packages\win32\lib\pywintypes.py:2: DeprecationWarn
    ing: the imp module is deprecated in favour of importlib; see the module's documentation for alterna
    tive uses
        import imp, sys, os
    
    -- Docs: https://docs.pytest.org/en/stable/warnings.html
    ================================== 20 passed, 1 warning in 4.51s ==================================
    

