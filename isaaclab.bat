@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Copyright (c) 2022-2024, The Isaac Lab Project Developers.
rem All rights reserved.
rem
rem SPDX-License-Identifier: BSD-3-Clause

rem Configurations
set "ISAACLAB_PATH=%~dp0"
goto main

rem Helper functions

rem extract the python from isaacsim
:extract_python_exe
rem Check if IsaacSim directory manually specified
rem Note: for manually build isaacsim, this: _build/linux-x86_64/release
if not "%ISAACSIM_PATH%"=="" (
    rem Use local build
    set build_path=%ISAACSIM_PATH%
) else (
    rem Use TeamCity build
    set build_path=%ISAACLAB_PATH%\_isaac_sim
)
rem check if using conda
if not "%CONDA_PREFIX%"=="" (
    rem use conda python
    set python_exe=%CONDA_PREFIX%\python
) else (
    rem check if isaacsim is installed
    pip show isaacsim-rl > nul 2>&1
    if errorlevel 1 (
        rem use python from kit if Isaac Sim not installed from pip
        set python_exe=%build_path%\python.bat
    ) else (
        rem use current python if Isaac Sim is installed from pip
        set "python_exe="
        for /f "delims=" %%i in ('where python') do (
            if not defined python_exe (
                set "python_exe=%%i"
            )
        )
    )
)
rem check if there is a python path available
if "%python_exe%"=="" (
    echo [ERROR] No python executable found at path: %build_path%
    exit /b 1
)
goto :eof


rem extract the simulator exe from isaacsim
:extract_isaacsim_exe
rem Check if IsaacSim directory manually specified
rem Note: for manually build isaacsim, this: _build\linux-x86_64\release
if not "%ISAACSIM_PATH%"=="" (
    rem Use local build
    set build_path=%ISAACSIM_PATH%
) else (
    rem Use TeamCity build
    set build_path=%ISAACLAB_PATH%\_isaac_sim
)
rem python executable to use
set isaacsim_exe=%build_path%\isaac-sim.bat
rem check if there is a python path available
if not exist "%isaacsim_exe%" (
    echo [ERROR] No isaac-sim executable found at path: %build_path%
    exit /b 1
)
goto :eof


rem check if input directory is a python extension and install the module
:install_isaaclab_extension
echo %ext_folder%
rem retrieve the python executable
call :extract_python_exe
rem if the directory contains setup.py then install the python module
if exist "%ext_folder%\setup.py" (
    echo     module: %ext_folder%
    call !python_exe! -m pip install --editable %ext_folder%
)
goto :eof


rem setup anaconda environment for Isaac Lab
:setup_conda_env
rem get environment name from input
set env_name=%conda_env_name%
rem check if conda is installed
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda could not be found. Please install conda and try again.
    exit /b 1
)
rem check if Isaac Sim directory manually specified
rem Note: for manually build Isaac Sim, this: _build\windows-x86_64\release
if not "%ISAACSIM_PATH%"=="" (
    rem Use local build
    set "build_path=%ISAACSIM_PATH%"
) else (
    rem Use TeamCity build
    set "build_path=%ISAACLAB_PATH%\_isaac_sim"
)
rem check if the environment exists
call conda env list | findstr /c:"%env_name%" >nul
if %errorlevel% equ 0 (
    echo [INFO] Conda environment named '%env_name%' already exists.
) else (
    echo [INFO] Creating conda environment named '%env_name%'...
    call conda create -y --name %env_name% python=3.10
)
rem cache current paths for later
set "cache_pythonpath=%PYTHONPATH%"
set "cache_ld_library_path=%LD_LIBRARY_PATH%"
rem clear any existing files
echo %CONDA_PREFIX%
del "%CONDA_PREFIX%\etc\conda\activate.d\setenv.bat" 2>nul
del "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv.bat" 2>nul
rem activate the environment
call conda activate %env_name%
rem setup directories to load isaac-sim variables
mkdir "%CONDA_PREFIX%\etc\conda\activate.d" 2>nul
mkdir "%CONDA_PREFIX%\etc\conda\deactivate.d" 2>nul
rem add variables to environment during activation
(
    echo @echo off
    rem for isaac-sim
    echo set CARB_APP_PATH=%build_path%\kit
    echo set EXP_PATH=%build_path%\apps
    echo set ISAAC_PATH=%build_path%
    echo set PYTHONPATH=%PYTHONPATH%;%build_path%\site
    echo set "RESOURCE_NAME=IsaacSim"
    echo doskey isaaclab=isaaclab.bat $*
) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat"
(
    echo $env:CARB_APP_PATH="%build_path%\kit"
    echo $env:EXP_PATH="%build_path%\apps"
    echo $env:ISAAC_PATH="%build_path%"
    echo $env:PYTHONPATH="%PYTHONPATH%;%build_path%\site"
    echo $env:RESOURCE_NAME="IsaacSim"
) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1"

rem reactivate the environment to load the variables
call conda activate %env_name%
rem remove variables from environment during deactivation
(
    echo @echo off
    echo rem for isaac-sim
    echo set "CARB_APP_PATH="
    echo set "EXP_PATH="
    echo set "ISAAC_PATH="
    echo set "RESOURCE_NAME="
    echo doskey isaaclab =
    echo.
    echo rem restore paths
    echo set "PYTHONPATH=%cache_pythonpath%"
    echo set "LD_LIBRARY_PATH=%cache_ld_library_path%"
) > "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv_vars.bat"
(
    echo $env:CARB_APP_PATH=""
    echo $env:EXP_PATH=""
    echo $env:ISAAC_PATH=""
    echo $env:RESOURCE_NAME=""
    echo $env:PYTHONPATH="%cache_pythonpath%"
    echo $env:LD_LIBRARY_PATH="%cache_pythonpath%"
) > "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv_vars.ps1"

rem install some extra dependencies
echo [INFO] Installing extra dependencies (this might take a few minutes)...
call conda install -c conda-forge -y importlib_metadata >nul 2>&1
rem deactivate the environment
call conda deactivate
rem add information to the user about alias
echo [INFO] Added 'isaaclab' alias to conda environment for 'isaaclab.bat' script.
echo [INFO] Created conda environment named '%env_name%'.
echo.
echo       1. To activate the environment, run:                conda activate %env_name%
echo       2. To install Isaac Lab extensions, run:            isaaclab -i
echo       4. To perform formatting, run:                      isaaclab -f
echo       5. To deactivate the environment, run:              conda deactivate
echo.
goto :eof


rem Update the vscode settings from template and Isaac Sim settings
:update_vscode_settings
echo [INFO] Setting up vscode settings...
rem Retrieve the python executable
call :extract_python_exe python_exe
rem Path to setup_vscode.py
set "setup_vscode_script=%ISAACLAB_PATH%\.vscode\tools\setup_vscode.py"
rem Check if the file exists before attempting to run it
if exist "%setup_vscode_script%" (
    call !python_exe! "%setup_vscode_script%"
) else (
    echo [WARNING] setup_vscode.py not found. Aborting vscode settings setup.
)
goto :eof


rem Print the usage description
:print_help
echo.
echo usage: %~nx0 [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-c] -- Utility to manage extensions in Isaac Lab.
echo.
echo optional arguments:
echo     -h, --help           Display the help content.
echo     -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks as extra dependencies. Default is 'all'.
echo     -f, --format         Run pre-commit to format the code and check lints.
echo     -p, --python         Run the python executable (python.bat) provided by Isaac Sim.
echo     -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
echo     -t, --test           Run all python unittest tests.
echo     -v, --vscode         Generate the VSCode settings file from template.
echo     -d, --docs           Build the documentation from source using sphinx.
echo     -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.
echo.
goto :eof


rem Main
:main

rem check argument provided
if "%~1"=="" (
    echo [Error] No arguments provided.
    call :print_help
    exit /b 1
)

rem pass the arguments
:loop
if "%~1"=="" goto :end
set "arg=%~1"

rem read the key
if "%arg%"=="-i" (
    rem install the python packages in omni.isaac.rl/source directory
    echo [INFO] Installing extensions inside the Isaac Lab repository...
    call :extract_python_exe
    for /d %%d in ("%ISAACLAB_PATH%\source\extensions\*") do (
        set ext_folder="%%d"
        call :install_isaaclab_extension
    )
    call !python_exe! -m pip show isaacsim-rl > nul 2>&1
    rem if not installing from pip, set up VScode
    if errorlevel 1 (
        rem setup vscode settings
        call :update_vscode_settings
    )
    rem install the python packages for supported reinforcement learning frameworks
    echo [INFO] Installing extra requirements such as learning frameworks...
    if "%~2"=="" (
        echo [INFO] Installing all rl-frameworks.
        set framework_name=all
    ) else if "%~2"=="none" (
        echo [INFO]  No rl-framework will be installed.
        set framework_name=none
        shift
    ) else (
        echo [INFO] Installing rl-framework: %2.
        set framework_name=%2
        shift
    )
    rem install the rl-frameworks specified
    !python_exe! -m pip install -e %ISAACLAB_PATH%\source\extensions\omni.isaac.lab_tasks[!framework_name!]
    shift
) else if "%arg%"=="--install" (
    rem install the python packages in omni.isaac.rl/source directory
    echo [INFO] Installing extensions inside the Isaac Lab repository...
    call :extract_python_exe
    for /d %%d in ("%ISAACLAB_PATH%\source\extensions\*") do (
        set ext_folder="%%d"
        call :install_isaaclab_extension
    )
    call !python_exe! -m pip show isaacsim-rl > nul 2>&1
    rem if not installing from pip, set up VScode
    if errorlevel 1 (
        rem setup vscode settings
        call :update_vscode_settings
    )
    rem install the python packages for supported reinforcement learning frameworks
    echo [INFO] Installing extra requirements such as learning frameworks...
    if "%~2"=="" (
        echo [INFO] Installing all rl-frameworks.
        set framework_name=all
    ) else if "%~2"=="none" (
        echo [INFO]  No rl-framework will be installed.
        set framework_name=none
        shift
    ) else (
        echo [INFO] Installing rl-framework: %2.
        set framework_name=%2
        shift
    )
    rem install the rl-frameworks specified
    !python_exe! -m pip install -e %ISAACLAB_PATH%\source\extensions\omni.isaac.lab_tasks[!framework_name!]
    shift
) else if "%arg%"=="-c" (
    rem use default name if not provided
    if not "%~2"=="" (
        echo [INFO] Using conda environment name: %2
        set conda_env_name=%2
        shift
    ) else (
        echo [INFO] Using default conda environment name: isaaclab
        set conda_env_name=isaaclab
    )
    call :setup_conda_env %conda_env_name%
    shift
) else if "%arg%"=="--conda" (
    rem use default name if not provided
    if not "%~2"=="" (
        echo [INFO] Using conda environment name: %2
        set conda_env_name=%2
        shift
    ) else (
        echo [INFO] Using default conda environment name: isaaclab
        set conda_env_name=isaaclab
    )
    call :setup_conda_env %conda_env_name%
    shift
) else if "%arg%"=="-f" (
    rem reset the python path to avoid conflicts with pre-commit
    rem this is needed because the pre-commit hooks are installed in a separate virtual environment
    rem and it uses the system python to run the hooks
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set cache_pythonpath=%PYTHONPATH%
        set PYTHONPATH=
    )

    rem run the formatter over the repository
    rem check if pre-commit is installed
    pip show pre-commit > nul 2>&1
    if errorlevel 1 (
        echo [INFO] Installing pre-commit...
        pip install pre-commit
    )

    rem always execute inside the Isaac Lab directory
    echo [INFO] Formatting the repository...
    pushd %ISAACLAB_PATH%
    call python -m pre_commit run --all-files
    popd >nul

    rem set the python path back to the original value
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set PYTHONPATH=%cache_pythonpath%
    )
    goto :end
) else if "%arg%"=="--format" (
    rem reset the python path to avoid conflicts with pre-commit
    rem this is needed because the pre-commit hooks are installed in a separate virtual environment
    rem and it uses the system python to run the hooks
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set cache_pythonpath=%PYTHONPATH%
        set PYTHONPATH=
    )

    rem run the formatter over the repository
    rem check if pre-commit is installed
    pip show pre-commit > nul 2>&1
    if errorlevel 1 (
        echo [INFO] Installing pre-commit...
        pip install pre-commit
    )

    rem always execute inside the Isaac Lab directory
    echo [INFO] Formatting the repository...
    pushd %ISAACLAB_PATH%
    call python -m pre_commit run --all-files
    popd >nul

    rem set the python path back to the original value
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set PYTHONPATH=%cache_pythonpath%
    )
    goto :end
) else if "%arg%"=="-p" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    echo [INFO] Using python from: !python_exe!
    REM Loop through all arguments - mimic shift
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! !allArgs!
    goto :end
) else if "%arg%"=="--python" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    echo [INFO] Using python from: !python_exe!
    REM Loop through all arguments - mimic shift
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! !allArgs!
    goto :end
) else if "%arg%"=="-s" (
    rem run the simulator exe provided by isaacsim
    call :extract_isaacsim_exe
    echo [INFO] Running isaac-sim from: %isaacsim_exe%
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !isaacsim_exe! --ext-folder %ISAACLAB_PATH%\source\extensions !allArgs1
    goto :end
) else if "%arg%"=="--sim" (
    rem run the simulator exe provided by Isaac Sim
    call :extract_isaacsim_exe
    echo [INFO] Running isaac-sim from: %isaacsim_exe%
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !isaacsim_exe! --ext-folder %ISAACLAB_PATH%\source\extensions !allArgs1
    goto :end
) else if "%arg%"=="-t" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! tools\run_all_tests.py !allArgs!
    goto :end
) else if "%arg%"=="--test" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! tools\run_all_tests.py !allArgs!
    goto :end
) else if "%arg%"=="-v" (
    rem update the vscode settings
    call :update_vscode_settings
    shift
    goto :end
) else if "%arg%"=="--vscode" (
    rem update the vscode settings
    call :update_vscode_settings
    shift
    goto :end
) else if "%arg%"=="-d" (
    rem build the documentation
    echo [INFO] Building documentation...
    call :extract_python_exe
    pushd %ISAACLAB_PATH%\docs
    call !python_exe! -m pip install -r requirements.txt >nul
    call !python_exe! -m sphinx -b html -d _build\doctrees . _build\html
    echo [INFO] To open documentation on default browser, run:
    echo xdg-open "%ISAACLAB_PATH%\docs\_build\html\index.html"
    popd >nul
    shift
    goto :end
) else if "%arg%"=="--docs" (
    rem build the documentation
    echo [INFO] Building documentation...
    call :extract_python_exe
    pushd %ISAACLAB_PATH%\docs
    call !python_exe! -m pip install -r requirements.txt >nul
    call !python_exe! -m sphinx -b html -d _build\doctrees . _build\html
    echo [INFO] To open documentation on default browser, run:
    echo xdg-open "%ISAACLAB_PATH%\docs\_build\html\index.html"
    popd >nul
    shift
    goto :end
) else if "%arg%"=="-h" (
    call :print_help
    goto :end
) else if "%arg%"=="--help" (
    call :print_help
    goto :end
) else (
    echo Invalid argument provided: %arg%
    call :print_help
    exit /b 1
)
goto loop

:end
exit /b 0
