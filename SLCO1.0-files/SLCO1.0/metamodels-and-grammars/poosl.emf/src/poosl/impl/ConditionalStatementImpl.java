/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

import poosl.ConditionalStatement;
import poosl.Expression;
import poosl.PooslPackage;
import poosl.Statement;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Conditional Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ConditionalStatementImpl#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.impl.ConditionalStatementImpl#getThenClause <em>Then Clause</em>}</li>
 *   <li>{@link poosl.impl.ConditionalStatementImpl#getElseClause <em>Else Clause</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ConditionalStatementImpl extends StatementImpl implements ConditionalStatement {
	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> condition;

	/**
	 * The cached value of the '{@link #getThenClause() <em>Then Clause</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getThenClause()
	 * @generated
	 * @ordered
	 */
	protected EList<Statement> thenClause;

	/**
	 * The cached value of the '{@link #getElseClause() <em>Else Clause</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getElseClause()
	 * @generated
	 * @ordered
	 */
	protected EList<Statement> elseClause;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ConditionalStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.CONDITIONAL_STATEMENT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getCondition() {
		if (condition == null) {
			condition = new EObjectResolvingEList<Expression>(Expression.class, this, PooslPackage.CONDITIONAL_STATEMENT__CONDITION);
		}
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Statement> getThenClause() {
		if (thenClause == null) {
			thenClause = new EObjectResolvingEList<Statement>(Statement.class, this, PooslPackage.CONDITIONAL_STATEMENT__THEN_CLAUSE);
		}
		return thenClause;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Statement> getElseClause() {
		if (elseClause == null) {
			elseClause = new EObjectResolvingEList<Statement>(Statement.class, this, PooslPackage.CONDITIONAL_STATEMENT__ELSE_CLAUSE);
		}
		return elseClause;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.CONDITIONAL_STATEMENT__CONDITION:
				return getCondition();
			case PooslPackage.CONDITIONAL_STATEMENT__THEN_CLAUSE:
				return getThenClause();
			case PooslPackage.CONDITIONAL_STATEMENT__ELSE_CLAUSE:
				return getElseClause();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PooslPackage.CONDITIONAL_STATEMENT__CONDITION:
				getCondition().clear();
				getCondition().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.CONDITIONAL_STATEMENT__THEN_CLAUSE:
				getThenClause().clear();
				getThenClause().addAll((Collection<? extends Statement>)newValue);
				return;
			case PooslPackage.CONDITIONAL_STATEMENT__ELSE_CLAUSE:
				getElseClause().clear();
				getElseClause().addAll((Collection<? extends Statement>)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case PooslPackage.CONDITIONAL_STATEMENT__CONDITION:
				getCondition().clear();
				return;
			case PooslPackage.CONDITIONAL_STATEMENT__THEN_CLAUSE:
				getThenClause().clear();
				return;
			case PooslPackage.CONDITIONAL_STATEMENT__ELSE_CLAUSE:
				getElseClause().clear();
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case PooslPackage.CONDITIONAL_STATEMENT__CONDITION:
				return condition != null && !condition.isEmpty();
			case PooslPackage.CONDITIONAL_STATEMENT__THEN_CLAUSE:
				return thenClause != null && !thenClause.isEmpty();
			case PooslPackage.CONDITIONAL_STATEMENT__ELSE_CLAUSE:
				return elseClause != null && !elseClause.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ConditionalStatementImpl
