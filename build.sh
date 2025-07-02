#!/bin/bash
# build.sh - Build script for Render deployment

echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

echo "Building frontend..."
cd frontend

# Install npm dependencies
echo "Installing frontend dependencies..."
npm install

# Build the frontend
echo "Running npm build..."
npm run build

# Check if build was successful
if [ -d "dist" ]; then
    echo "Frontend build successful!"
    echo "Contents of dist directory:"
    ls -la dist/
else
    echo "ERROR: Frontend build failed - dist directory not found!"
    exit 1
fi

cd ..

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo -e "\nCreating logs directory..."
mkdir -p logs

echo "Setting up database..."
# Create an empty database file
touch logs/users.db
echo "Database file created at logs/users.db"

echo -e "\nCreating templates directory if it doesn't exist..."
mkdir -p templates

# Create default templates if they don't exist
if [ ! -f "templates/wizard_prompt.txt" ]; then
    echo "Creating default wizard prompt template..."
    echo "You are a persuasive wizard. Your goal is: {{goal}}.
Engage the population agent in conversation." > templates/wizard_prompt.txt
fi

if [ ! -f "templates/judge_prompt.txt" ]; then
    echo "Creating default judge prompt template..."
    echo "# Instructions

You are an expert evaluator assessing conversations.

## Transcript to Evaluate

{{transcript}}

## Response Format

{{format_instructions}}" > templates/judge_prompt.txt
fi

if [ ! -f "templates/population_instruction.txt" ]; then
    echo "Creating default population instruction template..."
    echo "Generate {{n}} individuals. {{instruction}}
Return a JSON array where each object contains the fields 'name',
'personality', 'age', 'occupation', 'initial_goals' and 'memory_summary'." > templates/population_instruction.txt
fi

if [ ! -f "templates/self_improve_prompt.txt" ]; then
    echo "Creating default self improve prompt template..."
    echo "Analyze the following logs and suggest improvements:
{{logs}}" > templates/self_improve_prompt.txt
fi

echo "Build complete!"
echo "Final directory structure:"
ls -la
