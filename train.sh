#!/bin/bash
# WARNING: You MUST use bash to prevent errors

## Sun's Grid Engine parameters
# ... (see 'man qsub' for complete documentation)
# ... job name
#$ -N train_nmt
# ... make sure to use proper shell
#$ -S /bin/bash
# ... e-mail address to send notifications
#$ -M nidbhavsar989@gmail.com
# ... project name (use qconf -sprjl to watch project list)
#$ -P isometric_iswlt
# ... use current working directory for output
#$ -cwd

## Environment
# WARNING: your ~/.bashrc will NOT be loaded by SGE
# WARNING: do NOT load your default ~/.bashrc environment blindly; it will most likely break SGE!
# WARNING: include ONLY the SETSHELLs required for the job at hand; some SETSHELLs will break SGE!
# ... SETSHELL
. /idiap/resource/software/initfiles/shrc
SETSHELL <xyz>

## Job
<full-path-to>/my-program
