#! /bin/bash

# Developer Notice: We have used the latest synthcity vesion as of Jan 06, 2026. If reading this sounds old, you might want to
# change the SYNTHCITY_COMMIT_SHA to a more recent one.
# You can find a suitable one here: https://github.com/vanderschaarlab/synthcity/commits/main/
# Having said that, we cannot ensure compatibility with or reproducibility of our source code in such a case.
# Therefore, if you wish to completely reproduce our results, we highly recommend you stick to the same commit SHA.
# If you wish to run a new investigation leveraging the latest synthcity code, feel free to update the commit SHA and give it a try.

readonly SYNTHCITY_COMMIT_SHA='23f322fe381326ed01c41b13d469a06e38cce545'
readonly SYNTHCITY_TEMP_DIRECTORY='synthcity-base-temp'

echo "STEP 1 of 5: Cloning the base synthcity repository."
git clone https://github.com/vanderschaarlab/synthcity.git ${SYNTHCITY_TEMP_DIRECTORY} &> /dev/null \
    && echo -e "└❯ ✅ Successfully cloned the synthcity repo!\n" || echo -e "└❯ ❌ ERROR: Cloning synthcity failed!\n"


echo "STEP 2 of 5: Travelling back in time to ensure reproducibility. Using commit SHA" ${SYNTHCITY_COMMIT_SHA}
cd ${SYNTHCITY_TEMP_DIRECTORY}
git checkout ${SYNTHCITY_COMMIT_SHA} &> /dev/null \
    && echo -e "└❯ ✅ Successfully checked out to the commit SHA!\n" || echo -e "└❯ ❌ ERROR: Checking out the commit SHA failed!\n"


echo "STEP 3 of 5: Revamping synthcity"
git apply ../synthcity-patches/synthcity-351.patch &> /dev/null \
    && echo -e "└❯ ✅ Successfully applied PR 351 (Bugfix on PrivBayes)!" || echo -e "└❯ ❌ ERROR: Applying PR 351 has failed!"
git apply ../synthcity-patches/synthcity-353.patch &> /dev/null \
    && echo -e "└❯ ✅ Successfully applied PR 353 (Enable Pytorch 2.3+)!\n" || echo -e "└❯ ❌ ERROR: Applying PR 353 has failed!\n"


echo "STEP 4 of 5: Installing the revamped synthcity package"
# if you are not using uv, modify the following line accordingly, e.g., `pip install .` for pip-based package management
uv pip install . &> /dev/null \
    && echo -e "└❯ ✅ Successfully installed the revamped synthcity package!\n" || echo -e "└❯ ❌ ERROR: Installing the revamped synthcity package has failed!\n"


echo "STEP 5 of 5: Cleaning up"
cd ../ && rm -rf ${SYNTHCITY_TEMP_DIRECTORY} \
    && echo -e "└❯ ✅ Successfully cleaned up the temporary directory!\n" || echo -e "└❯ ❌ ERROR: Cleaning up the temporary directory has failed!\n"


echo "========= INFO =========\n"
echo "Installed synthcity:" $(uv pip show synthcity | grep -i version)
echo "Installed toch:" $(uv pip show torch | grep -i version)
