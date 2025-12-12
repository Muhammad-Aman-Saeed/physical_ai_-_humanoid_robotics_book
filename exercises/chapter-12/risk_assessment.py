"""
Risk assessment for robotic application exercise
File: exercises/chapter-12/risk_assessment.py

This exercise demonstrates how to perform a comprehensive risk assessment
for a robotic application, following safety standards and best practices.
It includes hazard identification, risk analysis, and risk evaluation.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json


class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class SafetyIntegrityLevel(Enum):
    SIL_1 = 1  # Low safety requirements
    SIL_2 = 2  # Medium safety requirements
    SIL_3 = 3  # High safety requirements
    SIL_4 = 4  # Very high safety requirements


@dataclass
class Hazard:
    """Represents a potential hazard in the robotic system"""
    id: str
    name: str
    description: str
    category: str  # Physical, Electrical, Software, Environmental
    potential_consequences: List[str]


@dataclass
class Risk:
    """Represents a risk with its attributes"""
    hazard_id: str
    severity: int  # 1-5 scale: 1=Minor, 5=Catastrophic
    probability: int  # 1-5 scale: 1=Rare, 5=Almost certain
    detectability: int  # 1-5 scale: 1=Almost impossible to detect, 5=Almost certain to detect
    initial_risk_level: RiskLevel
    residual_risk_level: RiskLevel
    mitigation_measures: List[str]


class RiskAssessmentTool:
    """
    A tool to perform comprehensive risk assessment for robotic applications
    """
    
    def __init__(self):
        self.hazards = []
        self.risks = []
        self.mitigation_measures = []
        
    def add_hazard(self, hazard: Hazard):
        """Add a hazard to the assessment"""
        self.hazards.append(hazard)
    
    def calculate_risk_level(self, severity: int, probability: int) -> RiskLevel:
        """Calculate risk level based on severity and probability"""
        risk_score = severity * probability
        
        if risk_score <= 4:
            return RiskLevel.LOW
        elif risk_score <= 8:
            return RiskLevel.MEDIUM
        elif risk_score <= 15:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def assess_risk(self, hazard_id: str, severity: int, probability: int, 
                   detectability: int, mitigation_measures: List[str] = None) -> Risk:
        """Assess a risk based on hazard and its parameters"""
        if mitigation_measures is None:
            mitigation_measures = []
        
        initial_risk_level = self.calculate_risk_level(severity, probability)
        
        # Apply mitigation measures to reduce risk (simplified model)
        # In practice, this would be more complex and specific to each measure
        if mitigation_measures:
            # Each mitigation measure reduces probability by 1 point (min 1)
            mitigated_probability = max(1, probability - len(mitigation_measures))
            residual_risk_level = self.calculate_risk_level(severity, mitigated_probability)
        else:
            residual_risk_level = initial_risk_level
        
        risk = Risk(
            hazard_id=hazard_id,
            severity=severity,
            probability=probability,
            detectability=detectability,
            initial_risk_level=initial_risk_level,
            residual_risk_level=residual_risk_level,
            mitigation_measures=mitigation_measures
        )
        
        self.risks.append(risk)
        return risk
    
    def get_required_sil(self) -> SafetyIntegrityLevel:
        """Determine required Safety Integrity Level based on highest risk"""
        if not self.risks:
            return SafetyIntegrityLevel.SIL_1
        
        max_risk_level = max(risk.initial_risk_level for risk in self.risks)
        
        if max_risk_level == RiskLevel.LOW:
            return SafetyIntegrityLevel.SIL_1
        elif max_risk_level == RiskLevel.MEDIUM:
            return SafetyIntegrityLevel.SIL_2
        elif max_risk_level == RiskLevel.HIGH:
            return SafetyIntegrityLevel.SIL_3
        else:
            return SafetyIntegrityLevel.SIL_4
    
    def generate_risk_matrix(self) -> np.ndarray:
        """Generate a risk matrix showing severity vs probability"""
        # Create a 5x5 matrix (severity x probability)
        matrix = np.zeros((5, 5))
        
        for risk in self.risks:
            # Convert to 0-indexed
            row = risk.severity - 1  # Severity as row (y-axis)
            col = risk.probability - 1  # Probability as column (x-axis)
            matrix[row, col] += 1
        
        return matrix
    
    def print_assessment_report(self):
        """Print a comprehensive risk assessment report"""
        print("=== ROBOTIC SYSTEM RISK ASSESSMENT REPORT ===\n")
        
        print("1. HAZARD IDENTIFICATION:")
        for hazard in self.hazards:
            print(f"   - {hazard.id}: {hazard.name}")
            print(f"     Description: {hazard.description}")
            print(f"     Category: {hazard.category}")
            print(f"     Consequences: {', '.join(hazard.potential_consequences)}\n")
        
        print("2. RISK ANALYSIS:")
        for risk in self.risks:
            hazard = next((h for h in self.hazards if h.id == risk.hazard_id), None)
            print(f"   Hazard: {hazard.name if hazard else risk.hazard_id}")
            print(f"   - Severity: {risk.severity}/5")
            print(f"   - Probability: {risk.probability}/5")
            print(f"   - Detectability: {risk.detectability}/5")
            print(f"   - Initial Risk Level: {risk.initial_risk_level.name}")
            print(f"   - Residual Risk Level: {risk.residual_risk_level.name}")
            print(f"   - Mitigation Measures: {', '.join(risk.mitigation_measures) if risk.mitigation_measures else 'None'}")
            print()
        
        print("3. SAFETY INTEGRITY LEVEL:")
        required_sil = self.get_required_sil()
        print(f"   Required SIL: {required_sil.name} ({required_sil.value})")
        
        print("\n4. RISK MATRIX:")
        risk_matrix = self.generate_risk_matrix()
        print("   Probability ->")
        print("   S  1  2  3  4  5")
        for i, row in enumerate(risk_matrix):
            print(f"   {i+1}  {int(row[0])}  {int(row[1])}  {int(row[2])}  {int(row[3])}  {int(row[4])}")
        print("   ^")
        print("   Severity")
        
        print("\n5. RECOMMENDATIONS:")
        high_risks = [r for r in self.risks if r.initial_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
        if high_risks:
            print("   CRITICAL: High or Very High risks identified that require immediate attention:")
            for risk in high_risks:
                hazard = next((h for h in self.hazards if h.id == risk.hazard_id), None)
                print(f"   - {hazard.name if hazard else risk.hazard_id}: Implement additional safeguards")
        else:
            print("   No critical risks identified.")


def create_robotic_system_example():
    """
    Create an example risk assessment for a humanoid robot in a home environment
    """
    print("Creating risk assessment for humanoid robot in home environment...")
    
    assessment = RiskAssessmentTool()
    
    # Add hazards for a humanoid robot
    hazards = [
        Hazard(
            id="HAZ-001",
            name="Physical Collision",
            description="Robot collides with humans or property causing injury or damage",
            category="Physical",
            potential_consequences=["Minor bruising", "Major injury", "Property damage"]
        ),
        Hazard(
            id="HAZ-002", 
            name="Falling Robot",
            description="Robot loses balance and falls, potentially causing injury to itself or others",
            category="Physical", 
            potential_consequences=["Robot damage", "Human injury", "Property damage"]
        ),
        Hazard(
            id="HAZ-003",
            name="Electrical Hazard",
            description="Faulty electrical components causing shock or fire",
            category="Electrical",
            potential_consequences=["Electric shock", "Fire", "System failure"]
        ),
        Hazard(
            id="HAZ-004",
            name="Unintended Movement",
            description="Robot performs unexpected movements due to software error",
            category="Software",
            potential_consequences=["Accidental collision", "Safety protocol bypass", "Loss of control"]
        ),
        Hazard(
            id="HAZ-005",
            name="Privacy Violation",
            description="Robot collects and transmits private information without consent",
            category="Security",
            potential_consequences=["Privacy breach", "Data theft", "Legal consequences"]
        ),
        Hazard(
            id="HAZ-006",
            name="Overheating",
            description="Motors or components overheat causing failure or fire",
            category="Physical",
            potential_consequences=["Component damage", "Fire", "Release of toxic fumes"]
        )
    ]
    
    for hazard in hazards:
        assessment.add_hazard(hazard)
    
    # Assess risks for each hazard
    # Format: assess_risk(hazard_id, severity, probability, detectability, mitigation_measures)
    assessment.assess_risk(
        "HAZ-001",  # Physical Collision
        severity=4,  # High severity if it happens
        probability=2,  # Medium probability with safety systems
        detectability=4,  # Can be detected with proximity sensors
        mitigation_measures=[
            "Proximity sensors", 
            "Speed limiting", 
            "Emergency stop", 
            "Collision avoidance algorithms"
        ]
    )
    
    assessment.assess_risk(
        "HAZ-002",  # Falling Robot
        severity=3,  # Medium to high severity
        probability=2,  # Medium probability in dynamic environments
        detectability=3,  # Can be detected with IMU and balance control
        mitigation_measures=[
            "Advanced balance control", 
            "Fall detection", 
            "Safe fall positioning", 
            "Obstacle detection"
        ]
    )
    
    assessment.assess_risk(
        "HAZ-003",  # Electrical Hazard
        severity=5,  # Very high severity
        probability=1,  # Low probability with proper design
        detectability=4,  # Can be detected with monitoring
        mitigation_measures=[
            "Regular electrical testing", 
            "Ground fault protection", 
            "Thermal monitoring", 
            "Certified components"
        ]
    )
    
    assessment.assess_risk(
        "HAZ-004",  # Unintended Movement
        severity=4,  # High severity
        probability=2,  # Medium probability with complex software
        detectability=3,  # Detectable through motion planning verification
        mitigation_measures=[
            "Formal verification of movement commands", 
            "Safety-rated controllers", 
            "Motion bounds checking", 
            "Redundant safety systems"
        ]
    )
    
    assessment.assess_risk(
        "HAZ-005",  # Privacy Violation
        severity=3,  # Medium severity for privacy
        probability=1,  # Low probability with proper security
        detectability=2,  # May be hard to detect immediately
        mitigation_measures=[
            "End-to-end encryption", 
            "User consent protocols", 
            "Local processing", 
            "Data minimization"
        ]
    )
    
    assessment.assess_risk(
        "HAZ-006",  # Overheating
        severity=4,  # High severity
        probability=2,  # Medium probability during extended operation
        detectability=4,  # Easily detected with temperature sensors
        mitigation_measures=[
            "Temperature monitoring", 
            "Thermal management", 
            "Duty cycle limiting", 
            "Automatic shutdown"
        ]
    )
    
    return assessment


def main():
    """
    Main function to run the risk assessment exercise
    """
    print("Starting risk assessment exercise for robotic application\n")
    
    # Create an example risk assessment
    assessment = create_robotic_system_example()
    
    # Print the assessment report
    assessment.print_assessment_report()
    
    # Additional analysis
    print("\n=== ADDITIONAL ANALYSIS ===")
    
    # Calculate overall risk metrics
    total_risks = len(assessment.risks)
    high_risks = len([r for r in assessment.risks if r.initial_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]])
    medium_risks = len([r for r in assessment.risks if r.initial_risk_level == RiskLevel.MEDIUM])
    low_risks = len([r for r in assessment.risks if r.initial_risk_level == RiskLevel.LOW])
    
    print(f"Total risks identified: {total_risks}")
    print(f"High/Very High risks: {high_risks}")
    print(f"Medium risks: {medium_risks}")
    print(f"Low risks: {low_risks}")
    
    print(f"\nRequired Safety Integrity Level: {assessment.get_required_sil().name}")
    
    print("\nRisk assessment exercise completed!")


if __name__ == "__main__":
    main()