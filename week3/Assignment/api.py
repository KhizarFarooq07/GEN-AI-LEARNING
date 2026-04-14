"""
FastAPI backend for the LinkedIn Content Curation System.

Provides REST endpoints:
- POST /plan - Create execution plan for a topic
- POST /execute - Execute the full content curation pipeline
"""

import asyncio
import json
import sys
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Task import (
        ContentCurationOrchestrator,
        ExecutionPlan,
        CurationResult,
        LinkedInPost,
    )
except ImportError as e:
    print(f"Warning: Could not import from Task.py: {e}")
    # These will be defined if imports fail
    ContentCurationOrchestrator = None
    ExecutionPlan = None
    CurationResult = None
    LinkedInPost = None


# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class PlanRequest(BaseModel):
    """Request model for creating an execution plan."""
    topic: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Recent trends in GenAI agents for backend engineers"
            }
        }


class PlanResponse(BaseModel):
    """Response model for execution plan."""
    topic: str
    steps: list
    created_at: str
    
    class Config:
        json_schema_extra = {
            "description": "Execution plan with dependency graph",
            "example": {
                "topic": "Topic",
                "created_at": "2026-04-13T11:33:41.061960",
                "steps": [
                    {
                        "step_id": 1,
                        "tool": "search",
                        "depends_on": [],
                        "status": "pending"
                    }
                ]
            }
        }


class ExecuteRequest(BaseModel):
    """Request model for executing content curation."""
    topic: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Recent trends in GenAI agents for backend engineers"
            }
        }


class PostData(BaseModel):
    """LinkedIn post data."""
    headline: str
    body: str
    hashtags: list
    call_to_action: Optional[str] = ""


class ExecuteResponse(BaseModel):
    """Response model for content curation result."""
    topic: str
    plan: Dict[str, Any]
    search_results: Optional[list]
    post: Optional[PostData]
    image_url: Optional[str]
    execution_log: list
    total_time: float
    
    class Config:
        json_schema_extra = {
            "description": "Complete content curation result",
            "example": {
                "topic": "Topic",
                "plan": {"steps": []},
                "search_results": [],
                "post": {
                    "headline": "Headline",
                    "body": "Body text",
                    "hashtags": ["AI", "Tech"],
                    "call_to_action": "Join the discussion"
                },
                "image_url": "https://...",
                "execution_log": [],
                "total_time": 30.5
            }
        }


class ExecutionStep(BaseModel):
    """Single step in execution plan."""
    step_id: int
    tool: str
    depends_on: list
    status: str = "pending"


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="LinkedIn Content Curation API",
    description="Agentic system for generating high-quality LinkedIn posts using planner-driven orchestration",
    version="1.0.0",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    global orchestrator
    print("\n[API] Initializing orchestrator...")
    orchestrator = ContentCurationOrchestrator()
    print("[API] Orchestrator initialized ✓\n")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LinkedIn Content Curation API"
    }


@app.post("/plan", response_model=PlanResponse, summary="Create Execution Plan")
async def create_plan(request: PlanRequest):
    """
    Create an execution plan for content curation.
    
    This endpoint analyzes the topic and creates a dependency-based execution plan
    showing which steps can run in parallel and which must run sequentially.
    
    **Parameters:**
    - `topic`: The content topic for curation (e.g., "GenAI trends in backend engineering")
    
    **Returns:**
    - Execution plan with steps and their dependencies
    - Example: Step 1 (Search) and Step 3 (ImageGen) run in parallel (no dependencies)
    - Step 2 (Summarize) depends on Step 1, Step 4 depends on Step 2, etc.
    
    **Example Request:**
    ```json
    {
        "topic": "Recent trends in GenAI agents for backend engineers"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "topic": "Recent trends in GenAI agents for backend engineers",
        "created_at": "2026-04-13T11:33:41.061960",
        "steps": [
            {
                "step_id": 1,
                "tool": "search",
                "depends_on": [],
                "params": {"topic": "...", "max_results": 5},
                "status": "pending"
            },
            {
                "step_id": 2,
                "tool": "summarize",
                "depends_on": [1],
                "params": {"topic": "..."},
                "status": "pending"
            },
            ...
        ]
    }
    ```
    """
    
    try:
        print(f"\n[API /plan] Creating plan for topic: '{request.topic}'")
        
        if not orchestrator:
            raise HTTPException(
                status_code=500,
                detail="Orchestrator not initialized"
            )
        
        # Create plan using planner agent
        plan = orchestrator.planner.analyze_intent(request.topic)
        
        response = PlanResponse(
            topic=plan.topic,
            steps=[s.to_dict() for s in plan.steps],
            created_at=plan.created_at
        )
        
        print(f"[API /plan] Plan created with {len(plan.steps)} steps ✓\n")
        
        return response
    
    except Exception as e:
        print(f"[API /plan] Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating plan: {str(e)}"
        )


@app.post("/execute", response_model=ExecuteResponse, summary="Execute Content Curation")
async def execute_curation(request: ExecuteRequest):
    """
    Execute the complete content curation pipeline.
    
    This endpoint orchestrates the entire agent system:
    1. Creates execution plan
    2. Executes steps based on dependencies
    3. Manages parallel execution where possible
    4. Returns final LinkedIn post and image
    
    **Execution Flow:**
    - Steps 1 (Search) & 3 (ImageGen) run in parallel
    - Step 2 (Summarize) waits for Step 1
    - Step 4 (GeneratePost) waits for Step 2
    - Step 5 (EditPost) waits for Step 4
    
    **Parameters:**
    - `topic`: The content topic for curation
    
    **Returns:**
    - Full execution plan with status and results
    - Generated LinkedIn post (headline, body, hashtags, CTA)
    - Generated image URL
    - Search results used for content curation
    - Execution log showing step-by-step progress
    - Total execution time
    
    **Example Request:**
    ```json
    {
        "topic": "How AI is transforming cloud infrastructure"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "topic": "How AI is transforming cloud infrastructure",
        "plan": {
            "steps": [
                {"step_id": 1, "tool": "search", "status": "completed", ...},
                ...
            ]
        },
        "search_results": [...],
        "post": {
            "headline": "...",
            "body": "...",
            "hashtags": ["AI", "Cloud"],
            "call_to_action": "..."
        },
        "image_url": "https://...",
        "execution_log": [
            "Starting execution of plan...",
            "Ready for parallel execution: steps [1, 3]",
            ...
        ],
        "total_time": 31.53
    }
    ```
    """
    
    try:
        print(f"\n[API /execute] Starting content curation for topic: '{request.topic}'")
        
        if not orchestrator:
            raise HTTPException(
                status_code=500,
                detail="Orchestrator not initialized"
            )
        
        # Execute the full pipeline
        result = await orchestrator.curate_content(request.topic)
        
        # Build response
        post_data = None
        if result.post:
            post_data = PostData(
                headline=result.post.headline,
                body=result.post.body,
                hashtags=result.post.hashtags,
                call_to_action=result.post.call_to_action
            )
        
        response = ExecuteResponse(
            topic=result.topic,
            plan=result.plan.to_dict(),
            search_results=result.search_results,
            post=post_data,
            image_url=result.image_url,
            execution_log=result.execution_log,
            total_time=result.total_time
        )
        
        print(f"[API /execute] Content curation completed in {result.total_time:.2f}s ✓\n")
        
        return response
    
    except Exception as e:
        print(f"[API /execute] Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing curation: {str(e)}"
        )


# ============================================================================
# DEBUG ENDPOINTS (Development only)
# ============================================================================

@app.get("/plan/{topic}", summary="Quick Plan Endpoint (Debug)")
async def quick_plan(topic: str):
    """
    Quick endpoint to create plan without JSON body.
    
    **Parameters:**
    - `topic`: URL query parameter with the topic
    
    **Example:**
    ```
    GET /plan/GenAI%20trends?
    ```
    """
    request = PlanRequest(topic=topic)
    return await create_plan(request)


@app.get("/execute/{topic}", summary="Quick Execute Endpoint (Debug)")
async def quick_execute(topic: str):
    """
    Quick endpoint to execute curation without JSON body.
    
    **Parameters:**
    - `topic`: URL query parameter with the topic
    
    **Example:**
    ```
    GET /execute/AI%20trends?
    ```
    """
    request = ExecuteRequest(topic=topic)
    return await execute_curation(request)


@app.get("/examples", summary="Get Example Topics")
async def get_examples():
    """Get example topics for testing."""
    return {
        "examples": [
            "Recent trends in GenAI agents for backend engineers",
            "How AI is transforming cloud infrastructure",
            "Best practices for building resilient microservices",
            "The future of serverless computing",
            "Machine learning in production systems"
        ]
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", summary="API Root")
async def root():
    """API root endpoint with documentation."""
    return {
        "name": "LinkedIn Content Curation API",
        "version": "1.0.0",
        "description": "Agentic system for generating high-quality LinkedIn posts",
        "endpoints": {
            "GET /health": "Health check",
            "POST /plan": "Create execution plan",
            "POST /execute": "Execute full content curation pipeline",
            "GET /examples": "Get example topics",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        }
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LINKEDIN CONTENT CURATION API - FASTAPI SERVER")
    print("="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
