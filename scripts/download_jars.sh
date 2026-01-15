#! /bin/bash

# The script requires the wget command-line utility.

readonly JARS_TARGET_DIRECTORY='../jars'

readonly HYFD_DOWNLOAD_LINK='https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/projekte/repeatability/DataProfiling/Metanome_Algorithms/HyFD-1.2-SNAPSHOT.jar'
readonly HYFD_JAR_FILE_NAME='HyFD.jar'

readonly METANOME_CLI_DOWNLOAD_LINK='https://github.com/sekruse/metanome-cli/releases/download/v1.1.0/metanome-cli-1.1.0.jar'
readonly METANOME_JAR_FILE_NAME='metanome-cli.jar'

# credits: https://hpi.de/naumann/projects/data-profiling-and-analytics/metanome-data-profiling/algorithms.html
echo "STEP 1 of 2: Downloading jar for HyFD."
(cd ${JARS_TARGET_DIRECTORY} && wget ${HYFD_DOWNLOAD_LINK} -O ${HYFD_JAR_FILE_NAME}&> /dev/null \
    && echo -e "└❯ ✅ Successfully downloaded the HyFD jar!\n" || echo -e "└❯ ❌ ERROR: Downloading the HyFD jar failed!\n" )

# credits: https://github.com/sekruse/metanome-cli
echo "STEP 2 of 2: Downloading jar for Metanome CLI."
(cd ${JARS_TARGET_DIRECTORY} && wget ${METANOME_CLI_DOWNLOAD_LINK} -O ${METANOME_JAR_FILE_NAME}&> /dev/null \
    && echo -e "└❯ ✅ Successfully downloaded the Metanome jar!\n" || echo -e "└❯ ❌ ERROR: Downloading the Metanome jar failed!\n" )

echo "========= INFO ========="
echo "The following jar files are now available:"
find ${JARS_TARGET_DIRECTORY} -name "*.jar"