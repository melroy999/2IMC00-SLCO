/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Assignment;
import poosl.Expression;
import poosl.PooslPackage;
import poosl.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Assignment</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.AssignmentImpl#getRightHandSide <em>Right Hand Side</em>}</li>
 *   <li>{@link poosl.impl.AssignmentImpl#getLeftHandSide <em>Left Hand Side</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class AssignmentImpl extends ExpressionImpl implements Assignment {
	/**
	 * The cached value of the '{@link #getRightHandSide() <em>Right Hand Side</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getRightHandSide()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> rightHandSide;

	/**
	 * The cached value of the '{@link #getLeftHandSide() <em>Left Hand Side</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getLeftHandSide()
	 * @generated
	 * @ordered
	 */
	protected Variable leftHandSide;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected AssignmentImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.ASSIGNMENT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getRightHandSide() {
		if (rightHandSide == null) {
			rightHandSide = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.ASSIGNMENT__RIGHT_HAND_SIDE);
		}
		return rightHandSide;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Variable getLeftHandSide() {
		if (leftHandSide != null && leftHandSide.eIsProxy()) {
			InternalEObject oldLeftHandSide = (InternalEObject)leftHandSide;
			leftHandSide = (Variable)eResolveProxy(oldLeftHandSide);
			if (leftHandSide != oldLeftHandSide) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.ASSIGNMENT__LEFT_HAND_SIDE, oldLeftHandSide, leftHandSide));
			}
		}
		return leftHandSide;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Variable basicGetLeftHandSide() {
		return leftHandSide;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setLeftHandSide(Variable newLeftHandSide) {
		Variable oldLeftHandSide = leftHandSide;
		leftHandSide = newLeftHandSide;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.ASSIGNMENT__LEFT_HAND_SIDE, oldLeftHandSide, leftHandSide));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.ASSIGNMENT__RIGHT_HAND_SIDE:
				return ((InternalEList<?>)getRightHandSide()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.ASSIGNMENT__RIGHT_HAND_SIDE:
				return getRightHandSide();
			case PooslPackage.ASSIGNMENT__LEFT_HAND_SIDE:
				if (resolve) return getLeftHandSide();
				return basicGetLeftHandSide();
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
			case PooslPackage.ASSIGNMENT__RIGHT_HAND_SIDE:
				getRightHandSide().clear();
				getRightHandSide().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.ASSIGNMENT__LEFT_HAND_SIDE:
				setLeftHandSide((Variable)newValue);
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
			case PooslPackage.ASSIGNMENT__RIGHT_HAND_SIDE:
				getRightHandSide().clear();
				return;
			case PooslPackage.ASSIGNMENT__LEFT_HAND_SIDE:
				setLeftHandSide((Variable)null);
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
			case PooslPackage.ASSIGNMENT__RIGHT_HAND_SIDE:
				return rightHandSide != null && !rightHandSide.isEmpty();
			case PooslPackage.ASSIGNMENT__LEFT_HAND_SIDE:
				return leftHandSide != null;
		}
		return super.eIsSet(featureID);
	}

} //AssignmentImpl
