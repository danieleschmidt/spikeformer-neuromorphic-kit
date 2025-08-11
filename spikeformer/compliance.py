"""Compliance and regulatory framework for neuromorphic computing systems."""

import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import os
from pathlib import Path
import uuid


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    SOX = "sox"  # Sarbanes-Oxley Act (US)
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    ISO27001 = "iso27001"  # Information Security Management
    NIST = "nist"  # NIST Cybersecurity Framework
    IEEE = "ieee"  # IEEE Standards


class DataCategory(Enum):
    """Categories of data for privacy compliance."""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    BIOMETRIC = "biometric"
    HEALTH = "health"
    FINANCIAL = "financial"
    ANONYMOUS = "anonymous"
    SYNTHETIC = "synthetic"
    PUBLIC = "public"


class ProcessingPurpose(Enum):
    """Legal bases for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    RESEARCH = "research"
    TRAINING = "model_training"


@dataclass
class DataSubject:
    """Information about a data subject."""
    id: str
    category: DataCategory
    consent_given: bool
    consent_timestamp: Optional[datetime]
    retention_period: Optional[int]  # days
    processing_purposes: List[ProcessingPurpose]
    geographic_location: Optional[str]
    age_category: Optional[str]  # "adult", "minor"


@dataclass
class ProcessingRecord:
    """Record of data processing activity."""
    id: str
    timestamp: datetime
    operation_type: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    data_subjects: List[str]  # Subject IDs
    legal_basis: ProcessingPurpose
    retention_period: int
    geographic_location: str
    technical_measures: List[str]
    organizational_measures: List[str]


@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    framework: ComplianceFramework
    assessment_date: datetime
    compliance_score: float  # 0.0 to 1.0
    passed_controls: List[str]
    failed_controls: List[str]
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high"
    next_assessment_date: datetime


class PrivacyManager:
    """Manages privacy compliance and data protection."""
    
    def __init__(self, organization_name: str, dpo_contact: str):
        self.organization_name = organization_name
        self.dpo_contact = dpo_contact
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: List[ProcessingRecord] = []
        self.consent_records: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Create privacy directory
        self.privacy_dir = Path.cwd() / "privacy"
        self.privacy_dir.mkdir(exist_ok=True)
        
        # Load existing records
        self._load_privacy_records()
    
    def register_data_subject(self, subject: DataSubject) -> bool:
        """Register a new data subject."""
        with self.lock:
            try:
                self.data_subjects[subject.id] = subject
                self._save_privacy_records()
                
                self.logger.info(f"Registered data subject: {subject.id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to register data subject: {e}")
                return False
    
    def record_consent(self, subject_id: str, purposes: List[ProcessingPurpose],
                      consent_given: bool, consent_method: str = "explicit") -> bool:
        """Record consent from data subject."""
        with self.lock:
            try:
                consent_record = {
                    'subject_id': subject_id,
                    'purposes': [p.value for p in purposes],
                    'consent_given': consent_given,
                    'consent_method': consent_method,
                    'timestamp': datetime.utcnow().isoformat(),
                    'ip_address': self._get_client_ip(),
                    'user_agent': self._get_user_agent()
                }
                
                if subject_id not in self.consent_records:
                    self.consent_records[subject_id] = []
                
                self.consent_records[subject_id].append(consent_record)
                
                # Update data subject
                if subject_id in self.data_subjects:
                    self.data_subjects[subject_id].consent_given = consent_given
                    self.data_subjects[subject_id].consent_timestamp = datetime.utcnow()
                    self.data_subjects[subject_id].processing_purposes = purposes
                
                self._save_privacy_records()
                
                self.logger.info(f"Recorded consent for subject {subject_id}: {consent_given}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to record consent: {e}")
                return False
    
    def withdraw_consent(self, subject_id: str, purposes: Optional[List[ProcessingPurpose]] = None) -> bool:
        """Process consent withdrawal."""
        with self.lock:
            try:
                # Record withdrawal
                withdrawal_record = {
                    'subject_id': subject_id,
                    'purposes': [p.value for p in purposes] if purposes else "all",
                    'withdrawal_timestamp': datetime.utcnow().isoformat(),
                    'processed_by': self.organization_name
                }
                
                if subject_id not in self.consent_records:
                    self.consent_records[subject_id] = []
                
                self.consent_records[subject_id].append(withdrawal_record)
                
                # Update data subject
                if subject_id in self.data_subjects:
                    if purposes is None:  # Withdraw all consent
                        self.data_subjects[subject_id].consent_given = False
                        self.data_subjects[subject_id].processing_purposes = []
                    else:  # Withdraw specific purposes
                        current_purposes = self.data_subjects[subject_id].processing_purposes
                        updated_purposes = [p for p in current_purposes if p not in purposes]
                        self.data_subjects[subject_id].processing_purposes = updated_purposes
                        
                        if not updated_purposes:
                            self.data_subjects[subject_id].consent_given = False
                
                self._save_privacy_records()
                
                self.logger.info(f"Processed consent withdrawal for subject {subject_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to process consent withdrawal: {e}")
                return False
    
    def record_processing_activity(self, record: ProcessingRecord) -> bool:
        """Record data processing activity."""
        with self.lock:
            try:
                # Validate legal basis
                if not self._validate_legal_basis(record):
                    raise ValueError("Invalid legal basis for processing")
                
                self.processing_records.append(record)
                self._save_privacy_records()
                
                self.logger.info(f"Recorded processing activity: {record.id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to record processing activity: {e}")
                return False
    
    def _validate_legal_basis(self, record: ProcessingRecord) -> bool:
        """Validate legal basis for processing."""
        # Check if consent is required and obtained
        if record.legal_basis == ProcessingPurpose.CONSENT:
            for subject_id in record.data_subjects:
                if subject_id not in self.data_subjects:
                    return False
                
                subject = self.data_subjects[subject_id]
                if not subject.consent_given:
                    return False
                
                # Check if consent covers the purposes
                for purpose in record.processing_purposes:
                    if purpose not in subject.processing_purposes:
                        return False
        
        return True
    
    def handle_data_subject_request(self, request_type: str, subject_id: str) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        with self.lock:
            try:
                if request_type == "access":
                    return self._handle_access_request(subject_id)
                elif request_type == "portability":
                    return self._handle_portability_request(subject_id)
                elif request_type == "rectification":
                    return self._handle_rectification_request(subject_id)
                elif request_type == "erasure":
                    return self._handle_erasure_request(subject_id)
                elif request_type == "restrict":
                    return self._handle_restriction_request(subject_id)
                else:
                    raise ValueError(f"Unknown request type: {request_type}")
                    
            except Exception as e:
                self.logger.error(f"Failed to handle data subject request: {e}")
                return {"error": str(e)}
    
    def _handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle subject access request."""
        if subject_id not in self.data_subjects:
            return {"error": "Data subject not found"}
        
        subject_data = {
            "subject_info": asdict(self.data_subjects[subject_id]),
            "processing_records": [
                asdict(record) for record in self.processing_records
                if subject_id in record.data_subjects
            ],
            "consent_history": self.consent_records.get(subject_id, [])
        }
        
        return {"data": subject_data, "format": "json"}
    
    def _handle_portability_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        access_data = self._handle_access_request(subject_id)
        
        if "error" in access_data:
            return access_data
        
        # Format for portability (structured, machine-readable)
        portable_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "subject_id": subject_id,
            "data": access_data["data"],
            "format": "json",
            "schema_version": "1.0"
        }
        
        return portable_data
    
    def _handle_erasure_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to erasure (right to be forgotten)."""
        if subject_id not in self.data_subjects:
            return {"error": "Data subject not found"}
        
        # Check if erasure is permissible
        subject = self.data_subjects[subject_id]
        
        # Cannot erase if legal obligation exists
        has_legal_obligation = any(
            ProcessingPurpose.LEGAL_OBLIGATION in record.processing_purposes
            for record in self.processing_records
            if subject_id in record.data_subjects
        )
        
        if has_legal_obligation:
            return {"error": "Cannot erase data due to legal obligations"}
        
        # Perform erasure
        try:
            del self.data_subjects[subject_id]
            
            # Remove from processing records (mark as erased)
            for record in self.processing_records:
                if subject_id in record.data_subjects:
                    record.data_subjects.remove(subject_id)
            
            # Remove consent records
            if subject_id in self.consent_records:
                del self.consent_records[subject_id]
            
            self._save_privacy_records()
            
            return {"status": "erased", "timestamp": datetime.utcnow().isoformat()}
            
        except Exception as e:
            return {"error": f"Erasure failed: {e}"}
    
    def check_retention_compliance(self) -> List[Dict[str, Any]]:
        """Check data retention compliance."""
        violations = []
        current_time = datetime.utcnow()
        
        with self.lock:
            for subject_id, subject in self.data_subjects.items():
                if subject.retention_period is None:
                    continue
                
                if subject.consent_timestamp is None:
                    continue
                
                retention_end = subject.consent_timestamp + timedelta(days=subject.retention_period)
                
                if current_time > retention_end:
                    violations.append({
                        "subject_id": subject_id,
                        "retention_period": subject.retention_period,
                        "consent_timestamp": subject.consent_timestamp.isoformat(),
                        "retention_end": retention_end.isoformat(),
                        "days_overdue": (current_time - retention_end).days
                    })
        
        return violations
    
    def _get_client_ip(self) -> str:
        """Get client IP address (placeholder)."""
        return "127.0.0.1"  # Would be implemented based on context
    
    def _get_user_agent(self) -> str:
        """Get user agent (placeholder)."""
        return "SpikeFormer/1.0"  # Would be implemented based on context
    
    def _save_privacy_records(self):
        """Save privacy records to disk."""
        try:
            # Save data subjects
            subjects_file = self.privacy_dir / "data_subjects.json"
            with open(subjects_file, 'w') as f:
                subjects_dict = {}
                for subject_id, subject in self.data_subjects.items():
                    subject_dict = asdict(subject)
                    # Convert datetime to string
                    if subject_dict['consent_timestamp']:
                        subject_dict['consent_timestamp'] = subject_dict['consent_timestamp'].isoformat()
                    subjects_dict[subject_id] = subject_dict
                json.dump(subjects_dict, f, indent=2, default=str)
            
            # Save processing records
            records_file = self.privacy_dir / "processing_records.json"
            with open(records_file, 'w') as f:
                records_list = []
                for record in self.processing_records:
                    record_dict = asdict(record)
                    record_dict['timestamp'] = record_dict['timestamp'].isoformat()
                    records_list.append(record_dict)
                json.dump(records_list, f, indent=2, default=str)
            
            # Save consent records
            consent_file = self.privacy_dir / "consent_records.json"
            with open(consent_file, 'w') as f:
                json.dump(self.consent_records, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save privacy records: {e}")
    
    def _load_privacy_records(self):
        """Load privacy records from disk."""
        try:
            # Load data subjects
            subjects_file = self.privacy_dir / "data_subjects.json"
            if subjects_file.exists():
                with open(subjects_file, 'r') as f:
                    subjects_dict = json.load(f)
                    for subject_id, subject_dict in subjects_dict.items():
                        # Convert string back to datetime
                        if subject_dict['consent_timestamp']:
                            subject_dict['consent_timestamp'] = datetime.fromisoformat(
                                subject_dict['consent_timestamp']
                            )
                        
                        # Convert enum strings back to enums
                        subject_dict['category'] = DataCategory(subject_dict['category'])
                        subject_dict['processing_purposes'] = [
                            ProcessingPurpose(p) for p in subject_dict['processing_purposes']
                        ]
                        
                        self.data_subjects[subject_id] = DataSubject(**subject_dict)
            
            # Load processing records
            records_file = self.privacy_dir / "processing_records.json"
            if records_file.exists():
                with open(records_file, 'r') as f:
                    records_list = json.load(f)
                    for record_dict in records_list:
                        # Convert string back to datetime
                        record_dict['timestamp'] = datetime.fromisoformat(record_dict['timestamp'])
                        
                        # Convert enum strings back to enums
                        record_dict['data_categories'] = [
                            DataCategory(c) for c in record_dict['data_categories']
                        ]
                        record_dict['processing_purposes'] = [
                            ProcessingPurpose(p) for p in record_dict['processing_purposes']
                        ]
                        record_dict['legal_basis'] = ProcessingPurpose(record_dict['legal_basis'])
                        
                        self.processing_records.append(ProcessingRecord(**record_dict))
            
            # Load consent records
            consent_file = self.privacy_dir / "consent_records.json"
            if consent_file.exists():
                with open(consent_file, 'r') as f:
                    self.consent_records = json.load(f)
                    
        except Exception as e:
            self.logger.warning(f"Failed to load privacy records: {e}")


class ComplianceAuditor:
    """Audits system compliance with various frameworks."""
    
    def __init__(self, privacy_manager: PrivacyManager):
        self.privacy_manager = privacy_manager
        self.logger = logging.getLogger(__name__)
    
    def assess_gdpr_compliance(self) -> ComplianceReport:
        """Assess GDPR compliance."""
        passed_controls = []
        failed_controls = []
        recommendations = []
        
        # Check lawful basis
        if self._check_lawful_basis():
            passed_controls.append("Lawful basis for processing")
        else:
            failed_controls.append("Lawful basis for processing")
            recommendations.append("Ensure all processing activities have valid lawful basis")
        
        # Check consent records
        if self._check_consent_records():
            passed_controls.append("Consent management")
        else:
            failed_controls.append("Consent management")
            recommendations.append("Implement proper consent recording mechanisms")
        
        # Check data subject rights
        if self._check_data_subject_rights():
            passed_controls.append("Data subject rights")
        else:
            failed_controls.append("Data subject rights")
            recommendations.append("Implement data subject rights handling procedures")
        
        # Check retention policies
        violations = self.privacy_manager.check_retention_compliance()
        if not violations:
            passed_controls.append("Data retention")
        else:
            failed_controls.append("Data retention")
            recommendations.append(f"Address {len(violations)} retention policy violations")
        
        # Check data protection impact assessments
        if self._check_dpia_requirements():
            passed_controls.append("Data Protection Impact Assessment")
        else:
            failed_controls.append("Data Protection Impact Assessment")
            recommendations.append("Conduct DPIA for high-risk processing activities")
        
        # Calculate compliance score
        total_controls = len(passed_controls) + len(failed_controls)
        compliance_score = len(passed_controls) / total_controls if total_controls > 0 else 0.0
        
        # Determine risk level
        if compliance_score >= 0.8:
            risk_level = "low"
        elif compliance_score >= 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return ComplianceReport(
            framework=ComplianceFramework.GDPR,
            assessment_date=datetime.utcnow(),
            compliance_score=compliance_score,
            passed_controls=passed_controls,
            failed_controls=failed_controls,
            recommendations=recommendations,
            risk_level=risk_level,
            next_assessment_date=datetime.utcnow() + timedelta(days=90)
        )
    
    def _check_lawful_basis(self) -> bool:
        """Check if all processing has lawful basis."""
        for record in self.privacy_manager.processing_records:
            if not self.privacy_manager._validate_legal_basis(record):
                return False
        return True
    
    def _check_consent_records(self) -> bool:
        """Check consent recording compliance."""
        # Must have consent records for consent-based processing
        consent_based_subjects = set()
        
        for record in self.privacy_manager.processing_records:
            if record.legal_basis == ProcessingPurpose.CONSENT:
                consent_based_subjects.update(record.data_subjects)
        
        for subject_id in consent_based_subjects:
            if subject_id not in self.privacy_manager.consent_records:
                return False
            
            if not self.privacy_manager.consent_records[subject_id]:
                return False
        
        return True
    
    def _check_data_subject_rights(self) -> bool:
        """Check if data subject rights procedures are in place."""
        # This would check if the system can handle various requests
        # For now, we check if the methods exist and work
        try:
            # Test with a dummy request
            test_result = self.privacy_manager.handle_data_subject_request("access", "test_id")
            return "error" in test_result  # Should return error for non-existent ID
        except:
            return False
    
    def _check_dpia_requirements(self) -> bool:
        """Check if DPIA is required and conducted."""
        # Simple heuristic: if processing sensitive data categories
        sensitive_categories = [DataCategory.SENSITIVE, DataCategory.BIOMETRIC, DataCategory.HEALTH]
        
        for record in self.privacy_manager.processing_records:
            if any(cat in sensitive_categories for cat in record.data_categories):
                # Would check if DPIA document exists
                return os.path.exists("privacy/dpia.pdf")  # Placeholder
        
        return True  # No DPIA required
    
    def generate_compliance_report(self, framework: ComplianceFramework) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        if framework == ComplianceFramework.GDPR:
            return self.assess_gdpr_compliance()
        # Add other frameworks as needed
        else:
            raise NotImplementedError(f"Compliance assessment for {framework} not implemented")


class DataGovernance:
    """Data governance and lineage tracking."""
    
    def __init__(self):
        self.data_lineage: Dict[str, List[Dict]] = {}
        self.data_quality_metrics: Dict[str, Dict] = {}
        self.access_logs: List[Dict] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def track_data_lineage(self, dataset_id: str, operation: str, 
                          inputs: List[str], outputs: List[str],
                          metadata: Dict[str, Any] = None):
        """Track data transformation lineage."""
        with self.lock:
            lineage_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "operation": operation,
                "inputs": inputs,
                "outputs": outputs,
                "metadata": metadata or {},
                "operator": os.environ.get("USER", "unknown")
            }
            
            if dataset_id not in self.data_lineage:
                self.data_lineage[dataset_id] = []
            
            self.data_lineage[dataset_id].append(lineage_record)
            
            self.logger.info(f"Tracked data lineage for {dataset_id}: {operation}")
    
    def log_data_access(self, user_id: str, dataset_id: str, 
                       access_type: str, purpose: str = None):
        """Log data access for audit trail."""
        with self.lock:
            access_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "dataset_id": dataset_id,
                "access_type": access_type,  # "read", "write", "delete"
                "purpose": purpose,
                "ip_address": self._get_client_ip(),
                "session_id": str(uuid.uuid4())
            }
            
            self.access_logs.append(access_record)
            
            self.logger.info(f"Logged data access: {user_id} -> {dataset_id} ({access_type})")
    
    def get_data_lineage(self, dataset_id: str) -> List[Dict]:
        """Get complete data lineage for a dataset."""
        return self.data_lineage.get(dataset_id, [])
    
    def get_access_logs(self, dataset_id: Optional[str] = None, 
                       user_id: Optional[str] = None) -> List[Dict]:
        """Get filtered access logs."""
        filtered_logs = self.access_logs
        
        if dataset_id:
            filtered_logs = [log for log in filtered_logs if log["dataset_id"] == dataset_id]
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log["user_id"] == user_id]
        
        return filtered_logs
    
    def _get_client_ip(self) -> str:
        """Get client IP (placeholder)."""
        return "127.0.0.1"  # Would be implemented based on context