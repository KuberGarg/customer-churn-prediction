pipeline {

    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'C:\\Users\\hp\\anaconda3\\python.exe -m pip install -r requirements.txt'
            }
        }

        stage('Run ML Project') {
            steps {
                bat 'C:\\Users\\hp\\anaconda3\\python.exe project.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t churn-project .'
            }
        }

    }
}