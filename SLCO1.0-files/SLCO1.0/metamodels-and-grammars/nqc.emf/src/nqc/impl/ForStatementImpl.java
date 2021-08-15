/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.Expression;
import nqc.ForStatement;
import nqc.NqcPackage;
import nqc.Statement;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>For Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.ForStatementImpl#getInitialization <em>Initialization</em>}</li>
 *   <li>{@link nqc.impl.ForStatementImpl#getCondition <em>Condition</em>}</li>
 *   <li>{@link nqc.impl.ForStatementImpl#getUpdate <em>Update</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ForStatementImpl extends ControlStructureImpl implements ForStatement {
	/**
	 * The cached value of the '{@link #getInitialization() <em>Initialization</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInitialization()
	 * @generated
	 * @ordered
	 */
	protected Statement initialization;

	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected Expression condition;

	/**
	 * The cached value of the '{@link #getUpdate() <em>Update</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUpdate()
	 * @generated
	 * @ordered
	 */
	protected Statement update;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ForStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getForStatement();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement getInitialization() {
		return initialization;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetInitialization(Statement newInitialization, NotificationChain msgs) {
		Statement oldInitialization = initialization;
		initialization = newInitialization;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.FOR_STATEMENT__INITIALIZATION, oldInitialization, newInitialization);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInitialization(Statement newInitialization) {
		if (newInitialization != initialization) {
			NotificationChain msgs = null;
			if (initialization != null)
				msgs = ((InternalEObject)initialization).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.FOR_STATEMENT__INITIALIZATION, null, msgs);
			if (newInitialization != null)
				msgs = ((InternalEObject)newInitialization).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.FOR_STATEMENT__INITIALIZATION, null, msgs);
			msgs = basicSetInitialization(newInitialization, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.FOR_STATEMENT__INITIALIZATION, newInitialization, newInitialization));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getCondition() {
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetCondition(Expression newCondition, NotificationChain msgs) {
		Expression oldCondition = condition;
		condition = newCondition;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.FOR_STATEMENT__CONDITION, oldCondition, newCondition);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCondition(Expression newCondition) {
		if (newCondition != condition) {
			NotificationChain msgs = null;
			if (condition != null)
				msgs = ((InternalEObject)condition).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.FOR_STATEMENT__CONDITION, null, msgs);
			if (newCondition != null)
				msgs = ((InternalEObject)newCondition).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.FOR_STATEMENT__CONDITION, null, msgs);
			msgs = basicSetCondition(newCondition, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.FOR_STATEMENT__CONDITION, newCondition, newCondition));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement getUpdate() {
		return update;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetUpdate(Statement newUpdate, NotificationChain msgs) {
		Statement oldUpdate = update;
		update = newUpdate;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.FOR_STATEMENT__UPDATE, oldUpdate, newUpdate);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setUpdate(Statement newUpdate) {
		if (newUpdate != update) {
			NotificationChain msgs = null;
			if (update != null)
				msgs = ((InternalEObject)update).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.FOR_STATEMENT__UPDATE, null, msgs);
			if (newUpdate != null)
				msgs = ((InternalEObject)newUpdate).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.FOR_STATEMENT__UPDATE, null, msgs);
			msgs = basicSetUpdate(newUpdate, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.FOR_STATEMENT__UPDATE, newUpdate, newUpdate));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.FOR_STATEMENT__INITIALIZATION:
				return basicSetInitialization(null, msgs);
			case NqcPackage.FOR_STATEMENT__CONDITION:
				return basicSetCondition(null, msgs);
			case NqcPackage.FOR_STATEMENT__UPDATE:
				return basicSetUpdate(null, msgs);
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
			case NqcPackage.FOR_STATEMENT__INITIALIZATION:
				return getInitialization();
			case NqcPackage.FOR_STATEMENT__CONDITION:
				return getCondition();
			case NqcPackage.FOR_STATEMENT__UPDATE:
				return getUpdate();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case NqcPackage.FOR_STATEMENT__INITIALIZATION:
				setInitialization((Statement)newValue);
				return;
			case NqcPackage.FOR_STATEMENT__CONDITION:
				setCondition((Expression)newValue);
				return;
			case NqcPackage.FOR_STATEMENT__UPDATE:
				setUpdate((Statement)newValue);
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
			case NqcPackage.FOR_STATEMENT__INITIALIZATION:
				setInitialization((Statement)null);
				return;
			case NqcPackage.FOR_STATEMENT__CONDITION:
				setCondition((Expression)null);
				return;
			case NqcPackage.FOR_STATEMENT__UPDATE:
				setUpdate((Statement)null);
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
			case NqcPackage.FOR_STATEMENT__INITIALIZATION:
				return initialization != null;
			case NqcPackage.FOR_STATEMENT__CONDITION:
				return condition != null;
			case NqcPackage.FOR_STATEMENT__UPDATE:
				return update != null;
		}
		return super.eIsSet(featureID);
	}

} //ForStatementImpl
