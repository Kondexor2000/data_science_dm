pipeline {
    agent any

    environment {
        IMAGE_NAME = 'ig-comments'
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Python syntax') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install -r requirements.txt
                    PYTHONPATH=src python -m compileall -q src
                '''
            }
        }
        stage('Build Docker image') {
            steps { sh 'docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .' }
        }
    }
    post {
        always { cleanWs() }
    }
}
