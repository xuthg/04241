"""REST API facade over the existing path-based tool layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.rest_jobs import LocalJobStore
from app.tools import (
    run_calibration_from_paths,
    run_data_analysis_from_paths,
    run_dataset_profile_from_paths,
    run_correction_from_paths,
    run_ensemble_from_paths,
    run_forecast_from_paths,
    run_hpo_from_paths,
    run_lifecycle_smoke_from_paths,
    run_model_asset_profile,
    run_risk_from_paths,
    run_train_model_bundle_from_paths,
    run_training_from_paths,
    run_warning_from_paths,
)
from typing import List, Dict, Any
import pandas as pd
import os
from datetime import datetime
import time
from fastapi import Body
import math


class PathToolRequest(BaseModel):
    dataset_path: str | None = None
    file_path: str | None = None
    output_root: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class OutputOptionsRequest(BaseModel):
    output_root: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ForecastItem(BaseModel):
    time: str
    precipitation: float
    potential_evapotranspiration: float
    streamflow: Optional[float] = None  # 允许为空（用于未来预测）


class ForecastRequest(BaseModel):
    data: List[ForecastItem]


class EnsembleItem(BaseModel):
    timestamp: Optional[str] = None
    forecast_xinanjiang: Optional[float] = None
    forecast_gr4j: Optional[float] = None
    forecast_rf: Optional[float] = None
    forecast_lstm: Optional[float] = None


class EnsembleRequest(BaseModel):
    data: List[EnsembleItem]
    method: Optional[str] = "weighted_mean"
    weights: Optional[List[float]] = None


class EnsembleResultItem(BaseModel):
    index: Optional[int] = None
    timestamp: Optional[str] = None
    ensemble_forecast: float


class CorrectionRequest(BaseModel):
    ensemble: List[EnsembleResultItem]  # 预测结果
    observation: List[ForecastItem]


class RiskItem(BaseModel):
    index: Optional[int] = None
    timestamp: Optional[str] = None
    ensemble_forecast: Optional[float] = None
    observed: Optional[float] = None
    corrected_forecast: float


class RiskRequest(BaseModel):
    data: List[RiskItem]
    thresholds: Optional[dict] = None


class WarningItem(BaseModel):
    index: Optional[int] = None
    timestamp: Optional[str] = None
    ensemble_forecast: Optional[float] = None
    observed: Optional[float] = None
    corrected_forecast: float
    risk_level: Optional[str] = None


class WarningRequest(BaseModel):
    data: List[WarningItem]
    warning_threshold: Optional[float] = 300.0
    lead_time_hours: Optional[int] = 24


class DatasetProfileRequest(BaseModel):
    data: List[Dict[str, Any]]


class TrainModelBundleRequest(BaseModel):
    data: List[ForecastItem]
    max_rows: Optional[int] = 12000
    sequence_length: Optional[int] = 8
    lstm_epochs: Optional[int] = 6


class ModelAssetsProfileRequest(BaseModel):
    bundle_path: Optional[str] = None


def create_app(*, job_root: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Huadong Hydro REST API", version="0.1.0")
    jobs = LocalJobStore(job_root)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "huadong-rest",
            "protocol": "rest",
            "mcp_compatibility": "preserved",
        }

    # 数据画像
    @app.post("/dataset/profile")
    def dataset_profile(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_dataset_profile_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    # 模型画像
    @app.post("/model-assets/profile")
    def model_assets_profile(request: OutputOptionssRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_model_asset_profile(
                output_root=request.output_root,
                options=request.options,
            )
        )

    # 模型训练，训练线性模型（linear），rf，lstm
    @app.post("/train-model-bundle")
    def train_model_bundle(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_train_model_bundle_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    # 预测，未来1小时
    @app.post("/forecast")
    def forecast(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_forecast_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    # 数据分析
    @app.post("/analysis")
    def analysis(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_data_analysis_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    # 预测结果集成，多个模型的结果合并为1个，采用bma方法
    @app.post("/ensemble")
    def ensemble(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_ensemble_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    @app.post("/correction")
    def correction(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_correction_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    @app.post("/risk")
    def risk(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_risk_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    @app.post("/warning")
    def warning(request: PathToolRequest) -> dict[str, Any]:
        return _run_sync(
            lambda: run_warning_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            )
        )

    @app.post("/training/jobs", status_code=202)
    def training_job(request: PathToolRequest) -> JSONResponse:
        return _submit_job(
            jobs,
            operation="training",
            input_payload=request.model_dump(mode="json"),
            runner=lambda: run_training_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            ),
        )

    @app.post("/calibration/jobs", status_code=202)
    def calibration_job(request: PathToolRequest) -> JSONResponse:
        return _submit_job(
            jobs,
            operation="calibration",
            input_payload=request.model_dump(mode="json"),
            runner=lambda: run_calibration_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            ),
        )

    @app.post("/hpo/jobs", status_code=202)
    def hpo_job(request: PathToolRequest) -> JSONResponse:
        return _submit_job(
            jobs,
            operation="hpo",
            input_payload=request.model_dump(mode="json"),
            runner=lambda: run_hpo_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            ),
        )

    @app.post("/lifecycle-smoke/jobs", status_code=202)
    def lifecycle_smoke_job(request: PathToolRequest) -> JSONResponse:
        return _submit_job(
            jobs,
            operation="lifecycle-smoke",
            input_payload=request.model_dump(mode="json"),
            runner=lambda: run_lifecycle_smoke_from_paths(
                dataset_path=request.dataset_path,
                file_path=request.file_path,
                output_root=request.output_root,
                options=request.options,
            ),
        )

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        payload = jobs.read(job_id)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "job_not_found",
                    "message": f"Unknown job_id: {job_id}",
                },
            )
        return payload

    @app.post("/forecast_api")
    def forecast_api(req: ForecastRequest = Body(...)):

        df = pd.DataFrame([item.dict() for item in req.data])

        # 👉 2. 写临时 CSV（模型必须要）
        base_dir = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong")
        )
        os.makedirs(base_dir, exist_ok=True)

        clear_dir = os.path.join(base_dir, "forecast")
        clear_old_files_recursive(clear_dir, hours=24)

        input_path = os.path.join(
            base_dir,
            f"input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        df.to_csv(input_path, index=False)

        # 👉 3. 调原始模型（不用改它）
        result = run_forecast_from_paths(
            file_path=input_path,
            output_root=base_dir,
            options={}
        )

        # 👉 4. 从结果里拿 CSV 路径
        run_dir = result["run_dir"]

        forecast_csv = os.path.join(run_dir, "forecast.csv")

        if not os.path.exists(forecast_csv):
            return {"error": "模型未生成forecast.csv"}

        # 👉 5. 读结果 CSV
        result_df = pd.read_csv(forecast_csv)

        # 👉 6. 转 JSON 返回
        return {
            "message": "预测成功",
            "data": result_df.to_dict(orient="records")
        }

    @app.post("/ensemble_api")
    def ensemble_api(req: EnsembleRequest):

        # 👉 转 DataFrame
        df = pd.DataFrame([item.dict() for item in req.data])

        if df.empty:
            return {"error": "data不能为空"}

        # 👉 临时保存 CSV（模型要求）
        base_dir = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong")
        )
        os.makedirs(base_dir, exist_ok=True)

        clear_dir = os.path.join(base_dir, "ensemble")
        clear_old_files_recursive(clear_dir, hours=24)

        input_path = os.path.join(
            base_dir,
            f"ensemble_input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )

        df.to_csv(input_path, index=False)

        # 👉 调原始方法
        result = run_ensemble_from_paths(
            file_path=input_path,
            output_root=base_dir,
            options={
                "method": req.method,
                "weights": req.weights
            }
        )

        # 👉 找输出 CSV
        run_dir = result["run_dir"]
        output_csv = os.path.join(run_dir, "ensemble.csv")

        if not os.path.exists(output_csv):
            return {"error": "未生成ensemble结果"}

        # 👉 读取结果
        result_df = pd.read_csv(output_csv)

        return {
            "message": "融合成功",
            "data": result_df.to_dict(orient="records")
        }

    @app.post("/correction_api")
    def correction_api(req: CorrectionRequest = Body(...)):

        if not req.ensemble or not req.observation:
            return {"error": "数据不能为空"}

        # 👉 基础目录（统一）
        base_dir = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong")
        )
        os.makedirs(base_dir, exist_ok=True)

        # 👉 清理 correction 目录
        clear_dir = os.path.join(base_dir, "correction")
        clear_old_files_recursive(clear_dir, hours=24)

        # ========================
        # 1️⃣ 写 ensemble CSV
        # ========================
        df_ens = pd.DataFrame([item.model_dump() for item in req.ensemble])

        ensemble_path = os.path.join(
            base_dir,
            f"correction_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        df_ens.to_csv(ensemble_path, index=False)

        # ========================
        # 2️⃣ 写 observation CSV
        # ========================
        df_obs = pd.DataFrame([item.model_dump() for item in req.observation])

        observation_path = os.path.join(
            base_dir,
            f"correction_obs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        df_obs.to_csv(observation_path, index=False)

        # ========================
        # 3️⃣ 调原方法（关键）
        # ========================
        result = run_correction_from_paths(
            file_path=ensemble_path,
            output_root=base_dir,
            options={
                "observation_dataset": observation_path,
                "observation_column": "streamflow"
            }
        )

        # ========================
        # 4️⃣ 读取结果
        # ========================
        run_dir = result["run_dir"]

        corrected_csv = os.path.join(run_dir, "corrected.csv")
        details_json = os.path.join(run_dir, "correction_details.json")

        if not os.path.exists(corrected_csv):
            return {"error": "未生成 corrected.csv"}

        result_df = pd.read_csv(corrected_csv)

        details = {}
        if os.path.exists(details_json):
            with open(details_json, "r", encoding="utf-8") as f:
                details = json.load(f)

        return {
            "message": "校正成功",
            "metrics": details.get("error_metrics"),
            "anomaly_info": details.get("anomaly_info"),
            "correction_summary": details.get("correction_summary"),
            "data": result_df.to_dict(orient="records")
        }

    @app.post("/risk_api")
    def risk_api(req: RiskRequest = Body(...)):

        if not req.data:
            return {"error": "data不能为空"}

        # 👉 基础目录
        base_dir = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong")
        )
        os.makedirs(base_dir, exist_ok=True)

        # 👉 清理 risk 目录
        clear_dir = os.path.join(base_dir, "risk")
        clear_old_files_recursive(clear_dir, hours=24)

        # ========================
        # 1️⃣ 写输入 CSV
        # ========================
        df = pd.DataFrame([item.model_dump() for item in req.data])

        input_path = os.path.join(
            base_dir,
            f"risk_input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )

        df.to_csv(input_path, index=False)

        # ========================
        # 2️⃣ 调原方法（不改核心）
        # ========================
        result = run_risk_from_paths(
            file_path=input_path,
            output_root=base_dir,
            options={
                "thresholds": req.thresholds or {
                    "flood": 300.0,
                    "severe": 500.0
                },
                "model_columns": ["corrected_forecast"]
            }
        )

        # ========================
        # 3️⃣ 读取结果
        # ========================
        run_dir = result["run_dir"]

        risk_json_path = os.path.join(run_dir, "risk.json")
        summary_path = os.path.join(run_dir, "summary.txt")

        if not os.path.exists(risk_json_path):
            return {"error": "未生成 risk.json"}

        with open(risk_json_path, "r", encoding="utf-8") as f:
            risk_data = json.load(f)

        summary_text = ""
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()

        # 👉 取原始数据
        timestamps = [item.timestamp for item in req.data]

        flood_probs = risk_data.get("exceed_prob_flood", [])
        severe_probs = risk_data.get("exceed_prob_severe", [])
        quantiles = risk_data.get("quantiles", {})

        P50 = quantiles.get("P50", [])

        # 👉 组装“可读结构”
        rows = []

        for i in range(len(timestamps)):
            row = {
                "index": i,
                "timestamp": timestamps[i],
                "exceed_prob_flood": flood_probs[i] if i < len(flood_probs) else None,
                "exceed_prob_severe": severe_probs[i] if i < len(severe_probs) else None,
                "P50": P50[i] if i < len(P50) else None
            }
            rows.append(row)

        return {
            "message": "风险分析完成",
            "thresholds": risk_data.get("thresholds"),
            "summary": summary_text,
            "data": rows
        }

    @app.post("/warning_api")
    def warning_api(req: WarningRequest = Body(...)):

        if not req.data:
            return {"error": "data不能为空"}

        # 👉 基础目录
        base_dir = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong")
        )
        os.makedirs(base_dir, exist_ok=True)

        # 👉 清理 warning 目录
        clear_dir = os.path.join(base_dir, "warning")
        clear_old_files_recursive(clear_dir, hours=24)

        # ========================
        # 1️⃣ 写输入 CSV
        # ========================
        df = pd.DataFrame([item.model_dump() for item in req.data])

        input_path = os.path.join(
            base_dir,
            f"warning_input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )

        df.to_csv(input_path, index=False)

        # ========================
        # 2️⃣ 调原方法
        # ========================
        result = run_warning_from_paths(
            file_path=input_path,
            output_root=base_dir,
            options={
                "forecast_column": "corrected_forecast",
                "warning_threshold": req.warning_threshold,
                "lead_time_hours": req.lead_time_hours
            }
        )

        # ========================
        # 3️⃣ 读取结果
        # ========================
        run_dir = result["run_dir"]

        warning_json = os.path.join(run_dir, "warning.json")
        summary_txt = os.path.join(run_dir, "summary.txt")

        if not os.path.exists(warning_json):
            return {"error": "未生成 warning.json"}

        with open(warning_json, "r", encoding="utf-8") as f:
            warning_data = json.load(f)

        summary = ""
        if os.path.exists(summary_txt):
            with open(summary_txt, "r", encoding="utf-8") as f:
                summary = f.read()

        # ========================
        # 4️⃣ 输出结构优化（核心）
        # ========================
        flood = warning_data.get("flood_warning", {})
        drought = warning_data.get("drought_warning", {})

        flood_level = flood.get("warning_level", "none")
        drought_level = drought.get("warning_level", "none")

        timestamps = [item.timestamp for item in req.data]

        rows = []

        for i, ts in enumerate(timestamps):

            # 👉 0/1 标志
            flood_flag = 1 if flood_level != "none" else 0
            drought_flag = 1 if drought_level != "none" else 0

            # 👉 综合等级
            if flood_flag == 1:
                level = "flood"
            elif drought_flag == 1:
                level = "drought"
            else:
                level = "normal"

            rows.append({
                "index": i,
                "timestamp": ts,
                "value": req.data[i].corrected_forecast,  # 👉 加这个很实用
                "flood_warning": flood_flag,
                "drought_warning": drought_flag,
                "warning_level": level
            })

        # ========================
        # 5️⃣ 返回
        # ========================
        return {
            "message": "预警分析完成",
            "threshold": req.warning_threshold,
            "summary": summary,

            # 👉 优化结构（前端用）
            "data": rows
        }

    @app.post("/dataset/profile_api")
    def dataset_profile_api(req: DatasetProfileRequest):

        if not req.data:
            return {"error": "data不能为空"}

        df = pd.DataFrame(req.data)

        if df.empty:
            return {"error": "数据为空"}

        # 👉 基本信息
        profile = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
        }

        # 👉 数值统计
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            profile["numeric_summary"] = df[numeric_cols].describe().to_dict()

        # 👉 时间字段检测（可选增强）
        time_cols = [col for col in df.columns if "time" in col.lower()]
        profile["time_columns"] = time_cols

        return {
            "message": "数据集分析完成",
            "profile": profile
        }

    @app.post("/train-model-bundle_api")
    def train_model_bundle_api(req: TrainModelBundleRequest = Body(...)):
        if not req.data:
            return {"error": "data不能为空"}

        base_dir = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong", "train")
        )
        os.makedirs(base_dir, exist_ok=True)

        df = pd.DataFrame([item.model_dump() for item in req.data])

        input_path = os.path.join(
            base_dir,
            f"train_input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        df.to_csv(input_path, index=False, encoding="utf-8")

        bundle_path = os.path.join(
            base_dir,
            "forecast_model_bundle.pt"
        )

        result = run_train_model_bundle_from_paths(
            file_path=input_path,
            output_root=base_dir,
            options={
                "bundle_path": bundle_path,
                "max_rows": req.max_rows,
                "sequence_length": req.sequence_length,
                "lstm_epochs": req.lstm_epochs,
            },
        )

        return {
            "message": "模型训练完成",
            "bundle_path": bundle_path,
            "result": result,
        }

    @app.post("/model-assets/profile_api")
    def model_assets_profile_api() -> dict[str, Any]:

        output_root = os.path.abspath(
            os.path.join("doc", "restful", "outputs", "huadong", "assets")
        )

        bundle_path = os.path.abspath(
            os.path.join(
                "doc", "restful", "outputs", "huadong", "train",
                "forecast_model_bundle.pt"
            )
        )

        result = _run_sync(
            lambda: run_model_asset_profile(
                output_root=output_root,
                options={
                    "bundle_path": bundle_path
                }
            )
        )

        try:
            artifact_paths = result.get("artifact_paths", [])

            profile_path = artifact_paths.get("model_assets_profile")

            if not profile_path:
                return {"error": "未找到模型资产结果"}

            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)

        except Exception as e:
            return {"error": f"解析失败: {str(e)}"}

        return {
            "message": "模型资产加载成功",
            "profile": profile
        }

    return app


def _run_sync(runner) -> dict[str, Any]:
    try:
        return runner()
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "not_found", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "invalid_request", "message": str(exc)},
        ) from exc


def clear_old_files_recursive(base_dir: str, hours: int = 24):
    now = time.time()
    expire_time = now - hours * 3600

    if not os.path.exists(base_dir):
        return

    for root, dirs, files in os.walk(base_dir, topdown=False):

        # 👉 删除文件
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if os.path.getmtime(file_path) < expire_time:
                    os.remove(file_path)
            except Exception:
                pass

        # 👉 删除空目录
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass


def _submit_job(
        jobs: LocalJobStore,
        *,
        operation: str,
        input_payload: dict[str, Any],
        runner,
) -> JSONResponse:
    try:
        job = jobs.submit(operation=operation, input_payload=input_payload, runner=runner)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "invalid_request", "message": str(exc)},
        ) from exc
    return JSONResponse(status_code=202, content=job)
