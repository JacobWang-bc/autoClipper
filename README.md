
# AutoClipper - Video and Audio Clipping Tool

AutoClipper is an AI-powered video and audio intelligent clipping tool that supports Azure Video Indexer and Gemini AI.

## 🚀 Docker Deployment (Recommended)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/autoClipper.git
cd autoClipper

# 2. Configure environment variables
cp .env.example .env  # Copy example configuration file
# Edit .env file with your API keys

# 3. Build and start
docker-compose build
docker-compose up -d

# 4. Access the application
# Local access: http://localhost:8089
```

### Common Commands

```bash
# Start service
docker-compose up -d

# Stop service
docker-compose down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart

# Rebuild
docker-compose build --no-cache
```

### Deployment Notes

- **Port Configuration**: Listens on port 8089 by default, can be modified in `docker-compose.yml`
- **Data Persistence**: Files are stored in the `./data` directory
- **Firewall**: Ensure server firewall allows access to the respective ports
- **Resource Requirements**: At least 2GB RAM recommended, video processing is resource-intensive
- **Network Access**: Ensure container can access Azure and Gemini APIs

## 📋 配置设置

FunClip支持Azure Video Indexer进行音视频处理。在项目根目录创建 `.env` 文件来配置认证凭据。

#### 🎯 Option 1: Free Trial Account (Recommended for new users)

```env
# Azure Video Indexer Trial Account Configuration
AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY=your-api-key
AZURE_VIDEO_INDEXER_ACCOUNT_ID=your-account-id
AZURE_VIDEO_INDEXER_LOCATION=trial

# Gemini API Key (for AI-powered clipping)
GOOGLE_AI_STUDIO_API_KEY=your-gemini-key
MAX_SEGMENTS=30
```

**Steps to get API Key:**
1. Visit https://api-portal.videoindexer.ai
2. Click Sign in at the top right to login with your Azure account
3. Go to Profile page and find Primary key or Secondary key
4. Copy the key and paste it into AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY

#### 💰 Option 2: Paid ARM Account

```env
# Azure ARM Paid Account Configuration
ARM_SUBSCRIPTION_ID=your-subscription-id
ARM_RESOURCE_GROUP=your-resource-group
ARM_ACCOUNT_NAME=your-account-name
AZURE_VIDEO_INDEXER_LOCATION=your-region

# Service Principal Authentication
AZURE_CLIENT_ID=your-client-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_SECRET=your-client-secret

# Gemini API Key
GOOGLE_AI_STUDIO_API_KEY=your-gemini-key
MAX_SEGMENTS=30
```

#### 🔑 Option 3: Manual Bearer Token

```env
# Manual Bearer Token
AZURE_VIDEO_INDEXER_BEARER_TOKEN=your-access-token

# Gemini API Key
GOOGLE_AI_STUDIO_API_KEY=your-gemini-key
MAX_SEGMENTS=30
```

#### 🔐 Web UI Authentication (Optional)

To protect your web interface with a login page, add these to your `.env` file:

```env
# Web UI Authentication
GRADIO_USERNAME=your-username
GRADIO_PASSWORD=your-password
```

If both `GRADIO_USERNAME` and `GRADIO_PASSWORD` are set, users must enter the correct credentials to access the application. If not set, the web UI will be accessible without authentication.

**Configuration Notes:**
- `.env` file contains sensitive information, do not commit it to Git
- Choose any one of the three options, the system will automatically select the appropriate method
- Trial account provides 10 hours of free processing time per month
- Gemini API key is used for AI-powered intelligent clipping features
- `MAX_SEGMENTS` controls the maximum number of video segments that can be displayed (default: 10)

## 🔧 Local Development

To run in a local development environment:

```bash

uv sync

# Start the application
python funclip/launch.py --listen --port 8080
```

## 🚨 Troubleshooting

### Docker Issues

**Build Failure**
```bash
# Clean Docker cache
docker system prune -f

# Rebuild
docker-compose build --no-cache
```

**Port Conflict**
- Check if other services are using port 8089
- Modify port mapping in `docker-compose.yml`

**Permission Issues**
- Ensure user has read/write permissions for `./data` and `./.gradio_temp` directories
- Check Docker Desktop permission settings

### Application Issues

**Cannot Access**
- Check if container is running: `docker-compose ps`
- Verify port mapping is correct
- Check firewall settings

**API Connection Failure**
- Confirm API keys in `.env` file are correct
- Check network connectivity to Azure and Gemini services
- View application logs: `docker-compose logs -f`

**File Upload Failure**
- Check `./.gradio_temp` directory permissions
- Confirm uploaded file formats are supported

## 📁 Project Structure

```
autoClipper/
├── funclip/                 # Main application code
│   ├── launch.py           # Application entry point
│   ├── azure_processor.py  # Azure integration
│   ├── videoclipper.py     # Video clipping logic
│   └── llm/               # AI functionality
├── data/                   # Data storage directory
├── .gradio_temp/          # Temporary files directory
├── Dockerfile             # Docker image configuration
├── docker-compose.yml    # Docker orchestration configuration
└── .env                   # Environment configuration (create manually)
```

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

<a name="Usage"></a>
