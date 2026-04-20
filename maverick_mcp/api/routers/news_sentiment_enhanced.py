"""
Enhanced news sentiment analysis using Finnhub (primary), Tiingo (fallback),
or LLM-based analysis on research tools.

This module provides reliable news sentiment analysis by:
1. Using Finnhub's news-sentiment + company-news endpoints (free tier, 60 req/min)
2. Falling back to Tiingo's get_news method (requires paid plan)
3. Falling back to LLM-based sentiment analysis using existing research tools
4. Never relying on undefined EXTERNAL_DATA_API_KEY

Finnhub is preferred because Tiingo's News API is a paid add-on (free-tier keys
get a 403 "You do not have permission to access the News API"). Finnhub's free
tier returns aggregate bullish/bearish percentages plus a buzz signal, which
maps cleanly onto the existing response shape.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from tiingo import TiingoClient

from maverick_mcp.api.middleware.mcp_logging import get_tool_logger
from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def get_finnhub_api_key() -> str | None:
    """Return the Finnhub API key if configured, else None."""
    key = os.getenv("FINNHUB_API_KEY")
    return key.strip() if key and key.strip() else None


def get_tiingo_client() -> TiingoClient | None:
    """Get or create Tiingo client if API key is available."""
    api_key = os.getenv("TIINGO_API_KEY")
    if api_key:
        try:
            config = {"session": True, "api_key": api_key}
            return TiingoClient(config)
        except Exception as e:
            logger.warning(f"Failed to initialize Tiingo client: {e}")
    return None


def get_llm():
    """Get LLM for sentiment analysis (optimized for speed)."""
    from maverick_mcp.providers.llm_factory import get_llm as get_llm_factory
    from maverick_mcp.providers.openrouter_provider import TaskType

    # Use sentiment analysis task type with fast preference
    return get_llm_factory(
        task_type=TaskType.SENTIMENT_ANALYSIS, prefer_fast=True, prefer_cheap=True
    )


async def _fetch_finnhub_company_news(
    ticker: str, start_date: datetime, end_date: datetime, api_key: str
) -> list[dict[str, Any]]:
    """
    Fetch recent company headlines from Finnhub.

    Finnhub returns a list of dicts shaped like:
        {
            "category": "company",
            "datetime": 1712332800,
            "headline": "...",
            "image": "...",
            "related": "AAPL",
            "source": "Reuters",
            "summary": "...",
            "url": "..."
        }

    We normalise each into the shape the downstream LLM / basic analyser
    expects (``title``, ``description``, ``publishedDate``, ``source``) so
    the rest of the pipeline is provider-agnostic.
    """
    params = {
        "symbol": ticker.upper(),
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "token": api_key,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{FINNHUB_BASE_URL}/company-news", params=params)
        resp.raise_for_status()
        raw = resp.json()

    if not isinstance(raw, list):
        return []

    normalised: list[dict[str, Any]] = []
    for article in raw:
        if not isinstance(article, dict):
            continue
        ts = article.get("datetime")
        published_iso = ""
        if ts is not None:
            try:
                published_iso = datetime.fromtimestamp(
                    int(ts), tz=timezone.utc
                ).isoformat()
            except (TypeError, ValueError, OSError):
                published_iso = ""
        normalised.append(
            {
                "title": article.get("headline", "") or "",
                "description": article.get("summary", "") or "",
                "publishedDate": published_iso,
                "source": article.get("source", "") or "Finnhub",
                "url": article.get("url", ""),
            }
        )
    return normalised


async def _fetch_finnhub_news_sentiment(
    ticker: str, api_key: str
) -> dict[str, Any] | None:
    """
    Fetch aggregate news sentiment from Finnhub.

    Response shape (non-premium tier, US equities):
        {
          "buzz": {"articlesInLastWeek": N, "buzz": float, "weeklyAverage": float},
          "companyNewsScore": float,          # 0..1
          "sectorAverageBullishPercent": float,
          "sectorAverageNewsScore": float,
          "sentiment": {"bearishPercent": float, "bullishPercent": float},
          "symbol": "AAPL"
        }

    Returns ``None`` on any error or on an empty/unparseable response (caller
    falls through to Tiingo / LLM fallback).
    """
    params = {"symbol": ticker.upper(), "token": api_key}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{FINNHUB_BASE_URL}/news-sentiment", params=params
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001 — any failure -> fallback
        logger.info(f"Finnhub news-sentiment fetch failed for {ticker}: {e}")
        return None

    if not isinstance(data, dict):
        return None
    sentiment = data.get("sentiment") or {}
    # Finnhub returns an empty dict (no error, no "sentiment" key populated)
    # for tickers it doesn't cover — treat that as "no signal".
    if not isinstance(sentiment, dict) or (
        "bullishPercent" not in sentiment and "bearishPercent" not in sentiment
    ):
        return None
    return data


def _summarise_finnhub_sentiment(
    finnhub_data: dict[str, Any], headlines: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Turn Finnhub's aggregate sentiment + recent headlines into the response
    shape the downstream chain directive consumes (same keys as the Tiingo
    path: ``sentiment``, ``confidence``, ``breakdown``, ``themes``,
    ``headlines``).
    """
    sentiment = finnhub_data.get("sentiment") or {}
    bullish = float(sentiment.get("bullishPercent") or 0.0)
    bearish = float(sentiment.get("bearishPercent") or 0.0)
    buzz = finnhub_data.get("buzz") or {}
    articles_in_week = int(buzz.get("articlesInLastWeek") or len(headlines) or 0)

    # Map percentages -> bullish/bearish/neutral. Finnhub's bullish+bearish
    # usually sums to ~1.0, so compare magnitudes directly with a small
    # neutrality band so balanced coverage doesn't flip on rounding noise.
    if bullish - bearish > 0.10:
        overall = "bullish"
    elif bearish - bullish > 0.10:
        overall = "bearish"
    else:
        overall = "neutral"

    # Confidence: strength of the directional skew, capped in [0, 1].
    confidence = min(1.0, max(0.0, abs(bullish - bearish)))

    # Breakdown: approximate article counts from percentages + total buzz.
    pos = int(round(bullish * articles_in_week))
    neg = int(round(bearish * articles_in_week))
    neu = max(0, articles_in_week - pos - neg)

    # Top headlines (headline strings, keep 3 like the Tiingo path).
    top_headlines = [h.get("title", "") for h in headlines[:3] if h.get("title")]

    # Themes: surface Finnhub's buzz + companyNewsScore + sector comparison so
    # the chain directive has concrete numbers to cite, not just keywords.
    news_score = finnhub_data.get("companyNewsScore")
    sector_bullish = finnhub_data.get("sectorAverageBullishPercent")
    weekly_avg = buzz.get("weeklyAverage")
    themes = [
        f"Bullish/bearish split: {bullish:.0%} / {bearish:.0%}",
    ]
    if news_score is not None:
        themes.append(f"Finnhub news score: {float(news_score):.2f}")
    if sector_bullish is not None:
        themes.append(
            f"Sector-average bullish: {float(sector_bullish):.0%}"
        )
    if weekly_avg:
        themes.append(
            f"Buzz: {articles_in_week} articles vs weekly avg {float(weekly_avg):.1f}"
        )

    return {
        "overall_sentiment": overall,
        "confidence": round(confidence, 3),
        "breakdown": {"positive": pos, "negative": neg, "neutral": neu},
        "themes": themes[:4],
        "headlines": top_headlines,
    }


async def get_news_sentiment_enhanced(
    ticker: str, timeframe: str = "7d", limit: int = 10
) -> dict[str, Any]:
    """
    Enhanced news sentiment analysis using Finnhub (primary), Tiingo (fallback),
    or LLM-based analysis on research tools.

    Provider order:
    1. Finnhub's ``/news-sentiment`` + ``/company-news`` (free tier, 60/min)
    2. Tiingo's ``get_news`` (paid add-on) + LLM-on-headlines analysis
    3. Research-based sentiment (``analyze_market_sentiment``)
    4. Neutral baseline fallback

    Args:
        ticker: Stock ticker symbol
        timeframe: Time frame for news (1d, 7d, 30d, etc.)
        limit: Maximum number of news articles to analyze

    Returns:
        Dictionary containing news sentiment analysis with confidence scores.
        Shape is stable across providers (ticker/sentiment/confidence/source/
        status/analysis/timeframe/request_id/timestamp) so chain directives
        don't need to branch on the underlying data source.
    """
    tool_logger = get_tool_logger("data_get_news_sentiment_enhanced")
    request_id = str(uuid.uuid4())

    try:
        # Calculate date range from timeframe (needed by all paths)
        end_date = datetime.now()
        days = int(timeframe.rstrip("d")) if timeframe.endswith("d") else 7
        start_date = end_date - timedelta(days=days)

        # Step 1: Try Finnhub (free tier covers both news-sentiment and
        # company-news for US equities). If FINNHUB_API_KEY is unset we skip
        # silently and fall through to Tiingo / LLM paths — same behaviour as
        # before, so the chain still gets a neutral response when no key is
        # configured rather than a hard failure.
        finnhub_key = get_finnhub_api_key()
        if finnhub_key:
            tool_logger.step(
                "finnhub_check", f"Checking Finnhub for {ticker}"
            )
            try:
                # Fetch aggregate sentiment + headlines concurrently; both are
                # independent GETs against the free tier.
                sentiment_task = asyncio.create_task(
                    _fetch_finnhub_news_sentiment(ticker, finnhub_key)
                )
                headlines_task = asyncio.create_task(
                    _fetch_finnhub_company_news(
                        ticker, start_date, end_date, finnhub_key
                    )
                )
                finnhub_sentiment, finnhub_headlines = await asyncio.wait_for(
                    asyncio.gather(
                        sentiment_task, headlines_task, return_exceptions=False
                    ),
                    timeout=12.0,
                )

                if finnhub_sentiment is not None:
                    summary = _summarise_finnhub_sentiment(
                        finnhub_sentiment, finnhub_headlines or []
                    )
                    tool_logger.complete(
                        f"Finnhub news sentiment completed for {ticker}"
                    )
                    return {
                        "ticker": ticker,
                        "sentiment": summary["overall_sentiment"],
                        "confidence": summary["confidence"],
                        "source": "finnhub_news_sentiment",
                        "status": "success",
                        "analysis": {
                            "articles_analyzed": len(finnhub_headlines or []),
                            "sentiment_breakdown": summary["breakdown"],
                            "key_themes": summary["themes"],
                            "recent_headlines": summary["headlines"][:3],
                        },
                        "timeframe": timeframe,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Aggregate sentiment missing (small/ADR/non-US ticker) but we
                # may still have headlines — run those through the LLM path so
                # the chain gets a real signal instead of neutral.
                if finnhub_headlines:
                    tool_logger.step(
                        "finnhub_llm_fallback",
                        f"Finnhub sentiment empty; running LLM on "
                        f"{len(finnhub_headlines)} headlines",
                    )
                    sentiment_result = await _analyze_news_sentiment_with_llm(
                        finnhub_headlines[:limit], ticker, tool_logger
                    )
                    tool_logger.complete(
                        f"Finnhub headlines + LLM completed for {ticker}"
                    )
                    return {
                        "ticker": ticker,
                        "sentiment": sentiment_result["overall_sentiment"],
                        "confidence": sentiment_result["confidence"],
                        "source": "finnhub_headlines_with_llm_analysis",
                        "status": "success",
                        "analysis": {
                            "articles_analyzed": len(finnhub_headlines),
                            "sentiment_breakdown": sentiment_result["breakdown"],
                            "key_themes": sentiment_result["themes"],
                            "recent_headlines": sentiment_result["headlines"][:3],
                        },
                        "timeframe": timeframe,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat(),
                    }

                tool_logger.step(
                    "finnhub_empty",
                    f"Finnhub returned no sentiment or headlines for {ticker}",
                )
            except TimeoutError:
                tool_logger.step(
                    "finnhub_timeout", "Finnhub timed out, trying Tiingo"
                )
            except httpx.HTTPStatusError as e:
                tool_logger.step(
                    "finnhub_http_error",
                    f"Finnhub HTTP {e.response.status_code}: falling through",
                )
            except Exception as e:  # noqa: BLE001 — any failure -> fallback
                tool_logger.step("finnhub_error", f"Finnhub error: {e}")

        # Step 2: Try Tiingo News API (paid add-on — will 403 on free tier,
        # which is caught below and falls through to research/LLM fallback).
        tool_logger.step("tiingo_check", f"Checking Tiingo News API for {ticker}")
        tiingo_client = get_tiingo_client()
        if tiingo_client:
            try:
                tool_logger.step(
                    "tiingo_fetch", f"Fetching news from Tiingo for {ticker}"
                )

                # Fetch news using Tiingo's get_news method
                news_articles = await asyncio.wait_for(
                    asyncio.to_thread(
                        tiingo_client.get_news,
                        tickers=[ticker],
                        startDate=start_date.strftime("%Y-%m-%d"),
                        endDate=end_date.strftime("%Y-%m-%d"),
                        limit=limit,
                        sortBy="publishedDate",
                        onlyWithTickers=True,
                    ),
                    timeout=10.0,
                )

                if news_articles:
                    tool_logger.step(
                        "llm_analysis",
                        f"Analyzing {len(news_articles)} articles with LLM",
                    )

                    # Analyze sentiment using LLM
                    sentiment_result = await _analyze_news_sentiment_with_llm(
                        news_articles, ticker, tool_logger
                    )

                    tool_logger.complete(
                        f"Tiingo news sentiment analysis completed for {ticker}"
                    )

                    return {
                        "ticker": ticker,
                        "sentiment": sentiment_result["overall_sentiment"],
                        "confidence": sentiment_result["confidence"],
                        "source": "tiingo_news_with_llm_analysis",
                        "status": "success",
                        "analysis": {
                            "articles_analyzed": len(news_articles),
                            "sentiment_breakdown": sentiment_result["breakdown"],
                            "key_themes": sentiment_result["themes"],
                            "recent_headlines": sentiment_result["headlines"][:3],
                        },
                        "timeframe": timeframe,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat(),
                    }

            except TimeoutError:
                tool_logger.step(
                    "tiingo_timeout", "Tiingo API timed out, using fallback"
                )
            except Exception as e:
                # Check if it's a permissions issue (free tier doesn't include news)
                if (
                    "403" in str(e)
                    or "permission" in str(e).lower()
                    or "unauthorized" in str(e).lower()
                ):
                    tool_logger.step(
                        "tiingo_no_permission",
                        "Tiingo news not available (requires paid plan)",
                    )
                else:
                    tool_logger.step("tiingo_error", f"Tiingo error: {str(e)}")

        # Step 3: Fallback to research-based sentiment
        tool_logger.step("research_fallback", "Using research-based sentiment analysis")

        from maverick_mcp.api.routers.research import analyze_market_sentiment

        # Use research tools to gather sentiment
        result = await asyncio.wait_for(
            analyze_market_sentiment(
                topic=f"{ticker} stock news sentiment recent {timeframe}",
                timeframe="1w" if days <= 7 else "1m",
                persona="moderate",
            ),
            timeout=15.0,
        )

        if result.get("success", False):
            sentiment_data = result.get("sentiment_analysis", {})
            return {
                "ticker": ticker,
                "sentiment": _extract_sentiment_from_research(sentiment_data),
                "confidence": sentiment_data.get("sentiment_confidence", 0.5),
                "source": "research_based_sentiment",
                "status": "fallback_success",
                "analysis": {
                    "overall_sentiment": sentiment_data.get("overall_sentiment", {}),
                    "key_themes": sentiment_data.get("sentiment_themes", [])[:3],
                    "market_insights": sentiment_data.get("market_insights", [])[:2],
                },
                "timeframe": timeframe,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Using research-based sentiment (Tiingo news unavailable on free tier)",
            }

        # Step 4: Basic neutral fallback
        return _provide_basic_sentiment_fallback(ticker, request_id)

    except Exception as e:
        tool_logger.error("sentiment_error", e, f"Sentiment analysis failed: {str(e)}")
        return _provide_basic_sentiment_fallback(ticker, request_id, str(e))


async def _analyze_news_sentiment_with_llm(
    news_articles: list, ticker: str, tool_logger
) -> dict[str, Any]:
    """Analyze news articles sentiment using LLM."""

    llm = get_llm()
    if not llm:
        # No LLM available, do basic analysis
        return _basic_news_analysis(news_articles)

    try:
        # Prepare news summary for LLM
        news_summary = []
        for article in news_articles[:10]:  # Limit to 10 most recent
            news_summary.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", "")[:200]
                    if article.get("description")
                    else "",
                    "publishedDate": article.get("publishedDate", ""),
                    "source": article.get("source", ""),
                }
            )

        # Create sentiment analysis prompt
        prompt = f"""Analyze the sentiment of these recent news articles about {ticker} stock.

News Articles:
{chr(10).join([f"- {a['title']} ({a['source']}, {a['publishedDate'][:10] if a['publishedDate'] else 'Unknown date'})" for a in news_summary[:5]])}

Provide a JSON response with:
1. overall_sentiment: "bullish", "bearish", or "neutral"
2. confidence: 0.0 to 1.0
3. breakdown: dict with counts of positive, negative, neutral articles
4. themes: list of 3 key themes from the news
5. headlines: list of 3 most important headlines

Response format:
{{"overall_sentiment": "...", "confidence": 0.X, "breakdown": {{"positive": X, "negative": Y, "neutral": Z}}, "themes": ["...", "...", "..."], "headlines": ["...", "...", "..."]}}"""

        # Get LLM analysis
        response = await asyncio.to_thread(lambda: llm.invoke(prompt).content)

        # Parse JSON response
        import json

        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            elif "{" in response:
                # Find JSON object in response
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            result = json.loads(json_str)

            # Ensure all required fields
            return {
                "overall_sentiment": result.get("overall_sentiment", "neutral"),
                "confidence": float(result.get("confidence", 0.5)),
                "breakdown": result.get(
                    "breakdown",
                    {"positive": 0, "negative": 0, "neutral": len(news_articles)},
                ),
                "themes": result.get(
                    "themes",
                    ["Market movement", "Company performance", "Industry trends"],
                ),
                "headlines": [a.get("title", "") for a in news_summary[:3]],
            }

        except (json.JSONDecodeError, ValueError) as e:
            tool_logger.step("llm_parse_error", f"Failed to parse LLM response: {e}")
            return _basic_news_analysis(news_articles)

    except Exception as e:
        tool_logger.step("llm_error", f"LLM analysis failed: {e}")
        return _basic_news_analysis(news_articles)


def _basic_news_analysis(news_articles: list) -> dict[str, Any]:
    """Basic sentiment analysis without LLM."""

    # Simple keyword-based sentiment
    positive_keywords = [
        "gain",
        "rise",
        "up",
        "beat",
        "exceed",
        "strong",
        "bull",
        "buy",
        "upgrade",
        "positive",
    ]
    negative_keywords = [
        "loss",
        "fall",
        "down",
        "miss",
        "below",
        "weak",
        "bear",
        "sell",
        "downgrade",
        "negative",
    ]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for article in news_articles:
        title = (
            article.get("title", "") + " " + article.get("description", "")
        ).lower()

        pos_score = sum(1 for keyword in positive_keywords if keyword in title)
        neg_score = sum(1 for keyword in negative_keywords if keyword in title)

        if pos_score > neg_score:
            positive_count += 1
        elif neg_score > pos_score:
            negative_count += 1
        else:
            neutral_count += 1

    total = len(news_articles)
    if total == 0:
        return {
            "overall_sentiment": "neutral",
            "confidence": 0.0,
            "breakdown": {"positive": 0, "negative": 0, "neutral": 0},
            "themes": [],
            "headlines": [],
        }

    # Determine overall sentiment
    if positive_count > negative_count * 1.5:
        overall = "bullish"
    elif negative_count > positive_count * 1.5:
        overall = "bearish"
    else:
        overall = "neutral"

    # Calculate confidence based on consensus
    max_count = max(positive_count, negative_count, neutral_count)
    confidence = max_count / total if total > 0 else 0.0

    return {
        "overall_sentiment": overall,
        "confidence": confidence,
        "breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
        },
        "themes": ["Recent news", "Market activity", "Company updates"],
        "headlines": [a.get("title", "") for a in news_articles[:3]],
    }


def _extract_sentiment_from_research(sentiment_data: dict) -> str:
    """Extract simple sentiment direction from research data."""

    overall = sentiment_data.get("overall_sentiment", {})

    # Check for sentiment keywords
    if isinstance(overall, dict):
        sentiment_str = str(overall).lower()
    else:
        sentiment_str = str(overall).lower()

    if "bullish" in sentiment_str or "positive" in sentiment_str:
        return "bullish"
    elif "bearish" in sentiment_str or "negative" in sentiment_str:
        return "bearish"

    # Check confidence for direction
    confidence = sentiment_data.get("sentiment_confidence", 0.5)
    if confidence > 0.6:
        return "bullish"
    elif confidence < 0.4:
        return "bearish"

    return "neutral"


def _provide_basic_sentiment_fallback(
    ticker: str, request_id: str, error_detail: str = None
) -> dict[str, Any]:
    """Provide basic fallback when all methods fail."""

    response = {
        "ticker": ticker,
        "sentiment": "neutral",
        "confidence": 0.0,
        "source": "fallback",
        "status": "all_methods_failed",
        "message": "Unable to fetch news sentiment - returning neutral baseline",
        "analysis": {
            "note": (
                "No news sentiment available. Set FINNHUB_API_KEY "
                "(free at finnhub.io, 60 req/min) to enable real "
                "bullish/bearish scoring; Tiingo News is a paid add-on."
            )
        },
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
    }

    if error_detail:
        response["error_detail"] = error_detail[:200]  # Limit error message length

    return response
