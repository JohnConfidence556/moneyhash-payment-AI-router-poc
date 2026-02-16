# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first
# This leverages Docker's cache to make builds faster
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir keeps the image small
# We increase the timeout to 1000 seconds to give slow connections a chance
RUN pip install --no-cache-dir --retries 10 --default-timeout=2000 -r requirements.txt

# 5. Copy the rest of your application code
# This copies your 'app/' folder (including the model)
COPY . .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the API
# We use uvicorn to serve the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]