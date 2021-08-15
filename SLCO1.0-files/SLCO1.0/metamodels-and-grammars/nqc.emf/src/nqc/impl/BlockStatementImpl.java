/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import java.util.Collection;

import nqc.BlockStatement;
import nqc.NqcPackage;
import nqc.Statement;
import nqc.Variable;

import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Block Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.BlockStatementImpl#getLocalVariables <em>Local Variables</em>}</li>
 *   <li>{@link nqc.impl.BlockStatementImpl#getStatements <em>Statements</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class BlockStatementImpl extends StatementImpl implements BlockStatement {
	/**
	 * The cached value of the '{@link #getLocalVariables() <em>Local Variables</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getLocalVariables()
	 * @generated
	 * @ordered
	 */
	protected EList<Variable> localVariables;

	/**
	 * The cached value of the '{@link #getStatements() <em>Statements</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStatements()
	 * @generated
	 * @ordered
	 */
	protected EList<Statement> statements;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BlockStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getBlockStatement();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getLocalVariables() {
		if (localVariables == null) {
			localVariables = new EObjectContainmentEList<Variable>(Variable.class, this, NqcPackage.BLOCK_STATEMENT__LOCAL_VARIABLES);
		}
		return localVariables;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Statement> getStatements() {
		if (statements == null) {
			statements = new EObjectContainmentEList<Statement>(Statement.class, this, NqcPackage.BLOCK_STATEMENT__STATEMENTS);
		}
		return statements;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.BLOCK_STATEMENT__LOCAL_VARIABLES:
				return ((InternalEList<?>)getLocalVariables()).basicRemove(otherEnd, msgs);
			case NqcPackage.BLOCK_STATEMENT__STATEMENTS:
				return ((InternalEList<?>)getStatements()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.BLOCK_STATEMENT__LOCAL_VARIABLES:
				return getLocalVariables();
			case NqcPackage.BLOCK_STATEMENT__STATEMENTS:
				return getStatements();
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
			case NqcPackage.BLOCK_STATEMENT__LOCAL_VARIABLES:
				getLocalVariables().clear();
				getLocalVariables().addAll((Collection<? extends Variable>)newValue);
				return;
			case NqcPackage.BLOCK_STATEMENT__STATEMENTS:
				getStatements().clear();
				getStatements().addAll((Collection<? extends Statement>)newValue);
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
			case NqcPackage.BLOCK_STATEMENT__LOCAL_VARIABLES:
				getLocalVariables().clear();
				return;
			case NqcPackage.BLOCK_STATEMENT__STATEMENTS:
				getStatements().clear();
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
			case NqcPackage.BLOCK_STATEMENT__LOCAL_VARIABLES:
				return localVariables != null && !localVariables.isEmpty();
			case NqcPackage.BLOCK_STATEMENT__STATEMENTS:
				return statements != null && !statements.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //BlockStatementImpl
