pipeline {

    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/KuberGarg/customer-churn-prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run ML Project') {
            steps {
                bat 'python project.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t churn-project .'
            }
        }

    }
}