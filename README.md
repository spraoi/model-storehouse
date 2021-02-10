
# generic-service

* Add Circleci link here

## Developing locally

Add these environment variables:

AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>


To start the generic service:

    docker-compose up api

To add or remove dependencies:
    NOTE: Be sure poetry is installed on your machine.  On Mac, use `brew install poetry`.

    poetry add <package_name>@~<x.x>
    poetry add requests@~2.24

    poetry remove <package_name>
    poetry remove requests

Testing the local code using mapped volume to the docker container:

    poetry install



### Docker Compose Commands
NOTE: All commands are executed from project root directory.

    COMPOSE_DOCKER_CLI_BUILD=1 docker-compose up [--build]

### Docker Build Commands
NOTE: All commands are executed from project root directory.

    docker build --target debug-image -t shaka_service .

## Branch Management

When starting any new development, branch from `dev` and open your pull requests
against `dev`. Try to avoid working in a long-lived feature branch. Small,
self-contained commits and pull requests make this easier. Sometimes you'll need
to set a configuration or feature flag to prevent incomplete features from being
included in code paths.


The `master` branch must be deploy-able at any time, which means that:
- We need to be reasonably sure that the merge is bug-free.
- Data layer changes should be forwards and backwards compatible.
- Any new operational dependencies should be met before merging.


# Deployment

- git checkout dev
- git pull --rebase
- git checkout -b <new_branch>
- git push --set-upstream origin <new_branch>
- Merge <new_branch> to dev using PR
- git checkout dev
- git pull --rebase
- Merge master into dev via PR (Only if dev was previously merged into master)
- git pull --rebase (dev branch)
- Merge dev into master via PR
