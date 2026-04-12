  @echo off
  REM ============================================================================
  REM MPI Matrix Multiplication - Build Script
  REM ============================================================================

  REM Get script directory
  for %%i in (%~dp0) do set SCRIPT_DIR=%%~fi

  echo ========================================
  echo MPI Matrix Multiplication - Build Script
  echo ========================================
  echo Source: %SCRIPT_DIR%\src
  echo Output: %SCRIPT_DIR%\bin
  echo Obj:    %SCRIPT_DIR%\obj
  echo ========================================

  REM Setup MSVC environment
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

  where cl >nul 2>&1
  if %ERRORLEVEL% neq 0 (
      echo Error: MSVC compiler not found
      echo Please install Visual Studio Build Tools
      exit /b 1
  )

  REM Create directories
  if not exist "%SCRIPT_DIR%\bin" mkdir "%SCRIPT_DIR%\bin"
  if not exist "%SCRIPT_DIR%\obj" mkdir "%SCRIPT_DIR%\obj"

  echo.
  echo [1/2] Building mpi_matrix_mul.exe...

  cl /O2 /EHsc /I"D:\Microsoft SDKs\MPI\Include" /Fo"%SCRIPT_DIR%\obj\mpi_matrix_mul.obj" /Fe"%SCRIPT_DIR%\bin\mpi_matrix_mul.exe" "%SCRIPT_DIR%\src\mpi_matrix_mul.c" "D:\Microsoft SDKs\MPI\Lib\x64\msmpi.lib"

  if %ERRORLEVEL% neq 0 (
      echo Build FAILED!
  ) else (
      echo Build SUCCESS: bin\mpi_matrix_mul.exe
  )

  echo.
  echo [2/2] Building serial_mat_mul.exe...

  cl /O2 /EHsc /Fo"%SCRIPT_DIR%\obj\serial_mat_mul.obj" /Fe"%SCRIPT_DIR%\bin\serial_mat_mul.exe" "%SCRIPT_DIR%\src\serial_mat_mul.c"

  if %ERRORLEVEL% neq 0 (
      echo Build FAILED!
  ) else (
      echo Build SUCCESS: bin\serial_mat_mul.exe
  )

  echo.
  echo ========================================
  echo Build Complete
  echo ========================================