
### Configuration Setup

FunClip supports Azure Video Indexer for audio/video processing. Create a `.env` file in the project root directory to configure authentication credentials.

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
USERNAME=your-username
PASSWORD=your-password
```

If both `USERNAME` and `PASSWORD` are set, users must enter the correct credentials to access the application. If not set, the web UI will be accessible without authentication.

**Configuration Notes:**
- `.env` file contains sensitive information, do not commit it to Git
- Choose any one of the three options, the system will automatically select the appropriate method
- Trial account provides 10 hours of free processing time per month
- Gemini API key is used for AI-powered intelligent clipping features
- `MAX_SEGMENTS` controls the maximum number of video segments that can be displayed (default: 10)


<a name="Usage"></a>
