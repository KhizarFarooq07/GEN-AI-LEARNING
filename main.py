"""
Gen AI Learning Path - Quick Experiment Runner

A utility script to run quick LLM experiments from the command line.
Supports both Groq and OpenAI APIs.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_groq_setup():
    """Verify that Groq is properly configured."""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found")
            print("   Setup: Add GROQ_API_KEY to .env file")
            return False
        
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=[{"role": "user", "content": "Say 'Hello, Groq!' in exactly two words."}],
            temperature=0,
            max_tokens=10
        )
        print("✓ Groq API connection successful")
        print(f"✓ Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Groq API failed: {e}")
        return False


def check_openai_setup():
    """Verify that OpenAI is properly configured."""
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            print("   Setup: Add OPENAI_API_KEY to .env file")
            return False
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, OpenAI!' in exactly two words."}],
            temperature=0,
            max_tokens=10
        )
        print("✓ OpenAI API connection successful")
        print(f"✓ Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False


def quick_temp_test_groq(prompt: str, temperatures: list = None):
    """Quick test with Groq."""
    if temperatures is None:
        temperatures = [0, 0.7, 1.0]
    
    from groq import Groq
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not configured")
        return
    
    client = Groq(api_key=api_key)
    
    print(f"\n📝 Prompt: {prompt}\n")
    print("=" * 60)
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp}")
        print("-" * 40)
        
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=80
        )
        
        print(response.choices[0].message.content)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("     🤖 Gen AI Learning Path - Quick Tester")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Check all APIs
        print("\n📋 Checking Groq setup...")
        groq_ok = check_groq_setup()
        
        print("\n📋 Checking OpenAI setup...")
        openai_ok = check_openai_setup()
        
        if not (groq_ok or openai_ok):
            print("\n⚠️  At least one API key is required!")
            sys.exit(1)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run a demo with Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Error: GROQ_API_KEY not configured")
            sys.exit(1)
        
        demo_prompt = "Complete: The secret to learning is..."
        quick_temp_test_groq(demo_prompt)
    
    else:
        # Show help
        print("\nUsage:")
        print("  python main.py test      - Check API setup")
        print("  python main.py demo      - Run demo experiment")
        print("\nAvailable notebooks:")
        print("  • week1/temperature_experiments_groq.ipynb  (Groq-based)")
        print("  • week1/temperature_experiments.ipynb       (OpenAI-based)")
        print("\nSetup instructions:")
        print("  1. Copy .env.example to .env")
        print("  2. Add API keys (Groq and/or OpenAI)")
        print("  3. Run: python main.py test")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

