# Hisaki_Py
A python code to analyze Hisaki data.

## How to install and use
1. Clone the Hisaki_Py repository to your computer (see, https://github.com/keimasunaga/Hisaki_Py).
2. Copy hskinit_example.py in docs to the directory where your code will be stored, and rename the file to hskinit.py.
3. Set path information in hskinit.py for your environment.
4. Copy cribsheets (written by jupyter notebooks) to the directory where your notebooks will be stored.
5. Run the cribsheets, and if successful, you are all set!

## Developer
Kei Masunaga (keimasunaga@gmail.com)

!## How to contribute
!If you want to contribute code, you can fork the repo and clone it from your repository. Otherwise you can directly clone it.
!Even if you do not plan to contribute code, raising issues of bugs, unexpected behavior, and suggestions for improvement would all be valuable.

## How to Contribute (via Fork and Pull Request)

1. Fork this repository to your own GitHub account by clicking the "Fork" button on the top right of the repository page.

2. Clone the forked repository to your local machine:

  ```bash
  git clone https://github.com/your-username/your-forked-repo.git
  cd your-forked-repo
  ```
3. Add the original repository as the upstream remote (only once):
  ```bash
  git remote add upstream https://github.com/original-owner/original-repo.git
  ```
4. Fetch and sync the latest code from the original repository:
  ```bash
  git checkout main
  git pull upstream main
  ```
5. Create a new branch based on the latest main:
  ```bash
  git checkout -b feature/your-feature-name
  ```
6. Make your changes, then stage and commit:
  ```bash
  git add .
  git commit -m "Describe your changes here"
  ```
7. Push the branch to your fork:
  ```bash
  git push origin feature/your-feature-name
  ```
8. Go to your fork on GitHub and open a Pull Request:
- Set the base repository to the original (main branch).
- Set the compare branch to your new feature branch.
- Write a clear title and description of your changes.
- Submit the Pull Request.

## License
N/A
